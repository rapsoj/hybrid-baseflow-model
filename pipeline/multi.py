"""Joint training and per-catchment evaluation across the configured chalk catchments."""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from . import data as D
from .config import load_config
from .ensemble import pool_quantiles
from .evaluate import evaluate_split
from .model import FlowTransformer, pinball_loss
from .multi_plots import (
    BASELINE, plot_by_horizon, plot_by_regime, plot_catchment_metrics, plot_random_predictions, plot_vs_baseline, variant_summary,
)
from .target import ANCHOR_MODES, seasonal_table, target_offset
from .baselines import climatology, persistence
from .evaluate import regime_metrics
from .target import seasonal_climatology
from .train import pick_device

logger = logging.getLogger(__name__)
HDE_SUFFIXES = ("rainfall_15min", "groundwaterLevel_15min", "waterFlow_15min")


MIN_FLOW = 1e-3  # m3/s
MIN_FLOWING_SHARE = 0.05


class _IneligibleCatchment(Exception):
    """Expected data-coverage reason for excluding one configured catchment."""


def _data_files_present(cfg: dict) -> bool:
    hde_dir = Path(cfg["paths"]["hde_dir"])
    return (
        all(list(hde_dir.glob(f"*_{suffix}.csv")) for suffix in HDE_SUFFIXES)
        and Path(cfg["paths"]["gefs_csv"]).is_file()
        and (
            D.geosmart_store.store_path(
                cfg["paths"]["geosmart_dir"], cfg["borehole"]["geosmart_prefix"]
            ).is_file()
            or bool(D.geosmart_store.loose_files(
                cfg["paths"]["geosmart_dir"], cfg["borehole"]["geosmart_prefix"]
            ))
        )
    )


def _prepare_catchment(cfg: dict) -> dict:
    hde = D.load_hde(cfg["paths"]["hde_dir"], cfg["data"]["step_minutes"], cfg["data"])
    gefs = D.load_gefs(cfg["paths"]["gefs_csv"], cfg["data"]["gefs"]["bin_hours"])
    geosmart = D.load_geosmart(
        cfg["paths"]["geosmart_dir"], cfg["borehole"]["geosmart_prefix"], cfg["data"]["geosmart"]
    )
    if not geosmart:
        raise _IneligibleCatchment("no usable GeoSmart ensemble member forecasts")
    samples = D.build_samples(cfg, hde, gefs, geosmart)
    common = samples.has_gefs & samples.has_geosmart
    samples = samples.subset(np.flatnonzero(common))
    d, dv = cfg["data"], cfg["derived"]
    gap = pd.Timedelta(minutes=d["step_minutes"]) * (dv["history_steps"] + dv["horizon_fine_steps"])
    split_idx = D.purged_chronological_split(
        samples.origins, gap, d["splits"]["train_fraction"], d["splits"]["val_fraction"]
    )
    minimum = d["splits"]["min_samples_per_split"]
    sizes = {k: len(v) for k, v in split_idx.items()}
    if min(sizes.values()) < minimum:
        raise _IneligibleCatchment(
            f"only {len(samples)} aligned samples, split sizes {sizes} (need {minimum} per split)"
        )
    D.assert_no_temporal_overlap(samples, split_idx)
    flowing = float((samples.subset(split_idx["train"]).y > MIN_FLOW).mean())
    if flowing < MIN_FLOWING_SHARE:
        raise _IneligibleCatchment(
            f"flow is ~zero in {1 - flowing:.1%} of training targets (ephemeral/dry gauge; baselines score zero loss)"
        )
    return {
        "cfg": cfg,
        "hde": hde,
        "samples": samples,
        "splits": {k: samples.subset(v) for k, v in split_idx.items()},
        "sizes": sizes,
    }


def _fit_joint(model, train_data, val_data, cfg: dict, device, use_geosmart: bool) -> dict:
    training = cfg["training"]
    opt = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    loader = DataLoader(TensorDataset(*train_data), batch_size=training["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=training["batch_size"])
    q = model.quantiles.to(device)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in loader:
            enc, dec, y, catchment_id, *members = batch
            if use_geosmart:
                geo, member_counts = members
                picks = (torch.rand(len(member_counts)) * member_counts).long()
                dec = dec.clone()
                dec[:, :, D.GEO_CH] = geo[torch.arange(len(member_counts)), picks]
            enc, dec, y, catchment_id = (
                enc.to(device), dec.to(device), y.to(device), catchment_id.to(device)
            )
            opt.zero_grad()
            loss = pinball_loss(model(enc, dec, catchment_id), y, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_data[0])

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for enc, dec, y, catchment_id in val_loader:
                enc, dec, y, catchment_id = (
                    enc.to(device), dec.to(device), y.to(device), catchment_id.to(device)
                )
                val_total += pinball_loss(model(enc, dec, catchment_id), y, q).item() * len(y)
        val_loss = val_total / len(val_data[0])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("joint epoch %3d  train %.4f  val %.4f", epoch, train_loss, val_loss)

        if val_loss < best - 1e-6:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= training["early_stopping_patience"]:
                logger.info("Joint training early stopped at epoch %d (best %d)", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Joint training did not produce a valid model state")
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best}


def _prepare_variant_data(states: dict, catchment_ids: dict, variant: dict, cfg: dict) -> tuple:
    mode, d = cfg["data"]["target_mode"], cfg["data"]
    channels = D.variant_dec_channels(variant["use_gefs"], variant["use_geosmart"], mode in ANCHOR_MODES)
    offsets, scalers, tensors, members, variant_splits = {}, {}, {}, {}, {}
    max_members = 0

    for key, state in states.items():
        split = variant_splits[key] = {
            name: D.geosmart_representation(
                samples, variant.get("geosmart_mode", "level"), variant.get("geosmart_ensemble", "members")
            ) if variant["use_geosmart"] else samples
            for name, samples in state["splits"].items()
        }
        table = seasonal_table(
            state["hde"], split["train"].window_end.max(), d["output_step_minutes"],
            cfg["model"]["quantiles"], d["seasonal_window_days"],
        )
        offsets[key] = {
            name: target_offset(mode, samples, table, cfg["model"]["quantiles"], d["output_step_minutes"])
            for name, samples in split.items()
        }
        scalers[key] = D.Scalers.fit(split["train"], offsets[key]["train"])
        tensors[key] = {
            name: tuple(torch.from_numpy(a) for a in scalers[key].transform(samples, channels, offsets[key][name]))
            for name, samples in split.items()
        }
        if variant["use_geosmart"]:
            members[key] = {
                name: scalers[key].transform_members(samples) for name, samples in split.items()
            }
            max_members = max(max_members, *(int(s.geo_n.max()) for s in split.values()))

    split_geo_n = {key: {name: s.geo_n for name, s in sp.items()} for key, sp in variant_splits.items()}

    def combine(name: str, expand_geo_val: bool = False):
        encs, decs, targets, ids = [], [], [], []
        geo_values, counts = [], []
        for key, state in states.items():
            enc, dec, target = tensors[key][name]
            cid = catchment_ids[key]
            if expand_geo_val and variant["use_geosmart"]:
                g, n = members[key][name], split_geo_n[key][name]
                k = cfg["ensemble"]["val_members_per_sample"]
                rng = np.random.default_rng(cfg["training"]["seed"] + cid)
                choices = np.floor(rng.random((len(n), k)) * n[:, None]).astype(int)
                for j in range(k):
                    dec_j = dec.numpy().copy()
                    dec_j[:, :, channels.index(D.GEO_CH)] = g[np.arange(len(n)), choices[:, j]]
                    decs.append(dec_j)
                    encs.append(enc.numpy())
                    targets.append(target.numpy())
                    ids.extend([cid] * len(n))
            else:
                encs.append(enc.numpy())
                decs.append(dec.numpy())
                targets.append(target.numpy())
                ids.extend([cid] * len(enc))
                if variant["use_geosmart"] and name == "train":
                    g = members[key][name]
                    padded = np.zeros((len(g), max_members, g.shape[-1]), dtype=np.float32)
                    padded[:, :g.shape[1]] = g
                    geo_values.append(padded)
                    counts.append(split_geo_n[key][name])
        arrays = [
            torch.from_numpy(np.concatenate(encs).astype(np.float32)),
            torch.from_numpy(np.concatenate(decs).astype(np.float32)),
            torch.from_numpy(np.concatenate(targets).astype(np.float32)),
            torch.tensor(ids, dtype=torch.long),
        ]
        if variant["use_geosmart"] and name == "train":
            arrays.extend([
                torch.from_numpy(np.concatenate(geo_values)),
                torch.from_numpy(np.concatenate(counts).astype(np.int64)),
            ])
        return tuple(arrays)

    train = combine("train")
    val = combine("val", expand_geo_val=True)
    return channels, scalers, offsets, tensors, members, train, val, variant_splits


def _build_model(cfg: dict, channels: list[int], n_catchments: int, skip_index: int | None = None) -> FlowTransformer:
    model, derived = cfg["model"], cfg["derived"]
    return FlowTransformer(
        enc_in=len(D.ENC_CHANNELS),
        dec_in=len(channels),
        history_steps=derived["history_steps"],
        n_decoder_tokens=derived["n_decoder_tokens"],
        n_output_tokens=derived["n_output_tokens"],
        token_steps=derived["token_steps"],
        quantiles=model["quantiles"],
        d_model=model["d_model"],
        n_heads=model["n_heads"],
        n_encoder_layers=model["n_encoder_layers"],
        n_decoder_layers=model["n_decoder_layers"],
        dim_feedforward=model["dim_feedforward"],
        dropout=model["dropout"],
        encoder_patch_steps=model["encoder_patch_steps"],
        n_catchments=n_catchments,
        geosmart_skip_index=skip_index,
    )


def _predict_catchment(model, scaler, channels, samples, tensors, members, offset, catchment_id,
                       use_geosmart, device, cfg, member_medians: list | None = None):
    enc, dec, _ = (item.numpy() for item in tensors)
    ids = torch.full((len(enc),), catchment_id, dtype=torch.long, device=device)
    model.eval()
    if not use_geosmart:
        predictions = []
        with torch.no_grad():
            for start in range(0, len(enc), 64):
                e = torch.from_numpy(enc[start:start + 64]).to(device)
                d = torch.from_numpy(dec[start:start + 64]).to(device)
                predictions.append(model(e, d, ids[start:start + 64]).cpu().numpy())
        return scaler.inverse_flow(np.concatenate(predictions), offset)

    geo_pos = channels.index(D.GEO_CH)
    pooled = []
    with torch.no_grad():
        for i, n in enumerate(samples.geo_n):
            n = int(n)
            member_dec = np.repeat(dec[i:i + 1], n, axis=0)
            member_dec[:, :, geo_pos] = members[i, :n]
            member_predictions = []
            for start in range(0, n, 64):
                e = torch.from_numpy(np.repeat(enc[i:i + 1], min(64, n - start), axis=0)).to(device)
                d = torch.from_numpy(member_dec[start:start + 64]).to(device)
                ids_batch = torch.full((len(d),), catchment_id, dtype=torch.long, device=device)
                member_predictions.append(model(e, d, ids_batch).cpu().numpy())
            flow_q = scaler.inverse_flow(np.concatenate(member_predictions), offset[i])
            if member_medians is not None:
                member_medians.append(flow_q[:, :, len(cfg["model"]["quantiles"]) // 2])
            pooled.append(pool_quantiles(flow_q, cfg["model"]["quantiles"], cfg["ensemble"]["pool_draws_per_member"]))
    return np.stack(pooled)


def _skill(model_loss: float, base_loss: float):
    return 1.0 - model_loss / base_loss if base_loss else None


def _full_metrics(state: dict, samples, prediction: np.ndarray, threshold: float) -> dict:
    """Overall, per-week and per-regime metrics plus skill vs climatology, seasonal climatology and persistence."""
    cfg = state["cfg"]
    d, dv = cfg["data"], cfg["derived"]
    quantiles = cfg["model"]["quantiles"]
    train = state["splits"]["train"]
    table = seasonal_table(
        state["hde"], train.window_end.max(), d["output_step_minutes"], quantiles, d["seasonal_window_days"]
    )
    args = (quantiles, dv["steps_per_day"], threshold, d["extreme"]["prediction_quantile"])
    y = samples.y
    metric = evaluate_split(y, prediction, *args)
    base_pred = {
        "climatology": climatology(train.y, len(samples), quantiles),
        "seasonal_climatology": seasonal_climatology(
            table, samples.origins, dv["horizon_steps"], d["output_step_minutes"]
        ),
        "persistence": persistence(samples.last_flow, dv["horizon_steps"], len(quantiles)),
    }
    regime_args = (quantiles, dv["steps_per_day"], samples.bfi, d["baseflow"]["bfi_threshold"])
    metric["by_regime"] = regime_metrics(y, prediction, *regime_args)
    metric["baselines"] = {}
    for name, pred in base_pred.items():
        base = evaluate_split(y, pred, *args)
        base["by_regime"] = regime_metrics(y, pred, *regime_args)
        metric["baselines"][name] = base
        metric[f"skill_vs_{name}"] = _skill(metric["quantile_loss"], base["quantile_loss"])
        for wk, wm in metric["by_week"].items():
            wm[f"skill_vs_{name}"] = _skill(wm["quantile_loss"], base["by_week"][wk]["quantile_loss"])
        for regime, rm in metric["by_regime"].items():
            for m, b in [(rm, base["by_regime"][regime])] + [
                (rm["by_week"][w], base["by_regime"][regime]["by_week"][w]) for w in rm["by_week"]
            ]:
                m[f"skill_vs_{name}"] = _skill(m["quantile_loss"], b["quantile_loss"]) if m["n_steps"] else None
    return metric


def select_catchments(config_path="config.yaml", overrides=()) -> tuple[dict, dict]:
    """Prepare every configured catchment: returns ({key: state} for the eligible ones, {key: reason} for the rest)."""
    base = load_config(config_path, "chilgrove", overrides)
    states, skipped = {}, {}
    for key in base["boreholes"]:
        cfg = load_config(config_path, key, overrides)
        if cfg["derived"] != base["derived"] or cfg["model"] != base["model"]:
            raise ValueError(f"Catchment {key} has incompatible model shape/configuration")
        if not _data_files_present(cfg):
            skipped[key] = "one or more required HDE, GEFS or GeoSmart files are absent"
            continue
        try:
            states[key] = _prepare_catchment(cfg)
        except _IneligibleCatchment as exc:
            skipped[key] = str(exc)
        except RuntimeError as exc:
            if not str(exc).startswith("No usable samples were found"):
                raise
            skipped[key] = str(exc)
    return states, skipped


def run_multi_catchment(config_path="config.yaml", overrides=(), run_id: str | None = None,
                        variant_names: list[str] | None = None) -> pd.DataFrame:
    """Train a shared catchment-embedded model on eligible local catchment data."""
    base = load_config(config_path, "chilgrove", overrides)
    torch.manual_seed(base["training"]["seed"])
    np.random.seed(base["training"]["seed"])
    device = pick_device(base["training"]["device"])
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")

    states, skipped = select_catchments(config_path, overrides)
    for key, reason in skipped.items():
        logger.info("Skipping %s: %s", key, reason)
    for key, state in states.items():
        logger.info("%s: eligible aligned samples %s", key, state["sizes"])

    if not states:
        raise RuntimeError("No catchments have complete local source files and enough aligned samples for all splits")
    catchment_keys = list(states)
    catchment_ids = {key: i for i, key in enumerate(catchment_keys)}
    models_root = Path(base["paths"]["models_dir"]) / "all_chalk" / run_id
    results_root = Path(base["paths"]["results_dir"]) / "all_chalk" / run_id
    results_root.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    test_predictions = {}
    rows = []

    variants = list(base["ablation"]["variants"])
    if variant_names is not None:
        by_name = {v["name"]: v for v in variants}
        missing = [n for n in variant_names if n not in by_name]
        if missing:
            raise ValueError(f"Unknown ablation variants {missing}; available: {list(by_name)}")
        variants = [by_name[n] for n in variant_names]
    for variant in variants:
        torch.manual_seed(base["training"]["seed"])
        channels, scalers, offsets, tensors, members, train, val, variant_splits = _prepare_variant_data(
            states, catchment_ids, variant, base
        )
        skip = channels.index(D.GEO_CH) if variant["use_geosmart"] and variant.get("geosmart_skip") else None
        model = _build_model(base, channels, len(catchment_keys), skip).to(device)
        logger.info(
            "[%s] training %d catchments, %d samples, %d parameters on %s",
            variant["name"], len(catchment_keys), len(train[0]), sum(p.numel() for p in model.parameters()), device,
        )
        started = time.time()
        fit = _fit_joint(model, train, val, base, device, variant["use_geosmart"])
        logger.info("[%s] finished in %.0fs", variant["name"], time.time() - started)

        variant_metrics = {}
        for key, state in states.items():
            cid = catchment_ids[key]
            for split_name in ("val", "test"):
                samples = variant_splits[key][split_name]
                member_medians = [] if split_name == "test" and variant["use_geosmart"] else None
                prediction = _predict_catchment(
                    model, scalers[key], channels, samples, tensors[key][split_name],
                    members.get(key, {}).get(split_name), offsets[key][split_name], cid,
                    variant["use_geosmart"], device, base, member_medians,
                )
                if split_name == "test":
                    test_predictions.setdefault(variant["name"], {})[key] = {
                        "origins": samples.origins, "truth": samples.y, "pred": prediction,
                        "members": member_medians, "n_members": samples.geo_n,
                    }
                threshold = D.historic_flow_threshold(
                    state["hde"], state["splits"]["train"].window_end.max(), base["data"]["extreme"]["quantile"]
                )
                metric = _full_metrics(state, samples, prediction, threshold)
                variant_metrics.setdefault(key, {})[split_name] = metric
            rows.append({
                "catchment": key,
                "variant": variant["name"],
                "n_train": state["sizes"]["train"],
                "n_val": state["sizes"]["val"],
                "n_test": state["sizes"]["test"],
                "quantile_loss": variant_metrics[key]["test"]["quantile_loss"],
                "median_mae": variant_metrics[key]["test"]["median_mae"],
                "coverage": variant_metrics[key]["test"][f"coverage_{base['model']['quantiles'][0]}_{base['model']['quantiles'][-1]}"],
            })
        all_metrics[variant["name"]] = variant_metrics

        model_dir = models_root / variant["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "derived": base["derived"], "data": base["data"], "model": base["model"],
                "training": base["training"], "variant": variant, "n_catchments": len(catchment_keys),
            },
            "catchment_keys": catchment_keys,
            "catchment_ids": catchment_ids,
            "scalers": {key: scaler.to_dict() for key, scaler in scalers.items()},
            "decoder_channels": [D.DEC_CHANNELS[c] for c in channels],
        }, model_dir / "model.pt")
        pd.DataFrame(fit["history"]).to_csv(model_dir / "training_history.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_root / "catchment_metrics.csv", index=False)
    with open(results_root / "metrics.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "seed": base["training"]["seed"],
            "catchment_keys": catchment_keys,
            "split_sizes": {key: states[key]["sizes"] for key in catchment_keys},
            "metrics": all_metrics,
        }, f, indent=2)
    with open(results_root / "config.yaml", "w") as f:
        yaml.safe_dump({
            "catchments": catchment_keys,
            "catchment_ids": catchment_ids,
            "data": base["data"],
            "model": base["model"],
            "training": base["training"],
            "variants": variants,
        }, f, sort_keys=False)
    summary = variant_summary(comparison)
    summary.to_csv(results_root / "variant_summary.csv", index=False)
    logger.info("Variant summary (test quantile loss relative to %s)\n%s", BASELINE, summary.to_string(index=False))
    plot_path = plot_catchment_metrics(comparison, results_root / "catchment_metrics.png")
    plot_vs_baseline(comparison, results_root / "variants_vs_baseline.png")
    plot_by_horizon(all_metrics, results_root / "metrics_by_horizon.png")
    plot_by_regime(all_metrics, results_root / "metrics_by_regime.png")
    plot_random_predictions(
        test_predictions, base["model"]["quantiles"], base["data"]["output_step_minutes"] / 60,
        results_root / "random_predictions", base["training"]["seed"],
        with_variant=next(v["name"] for v in variants if v["use_geosmart"]),
    )
    logger.info("Results -> %s | weights -> %s | plot -> %s", results_root, models_root, plot_path)
    return comparison
