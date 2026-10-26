"""Training / ablation orchestration: build samples, split, train each variant, evaluate, persist."""
import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from . import data as D
from .baselines import climatology, persistence
from .config import snapshot
from .ensemble import expand_validation, predict_flow
from .evaluate import evaluate_split, regime_metrics
from .target import ANCHOR_MODES, seasonal_climatology, seasonal_table, target_offset
from .model import FlowTransformer, pinball_loss

logger = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict, enc_in: int, dec_in: int, geosmart_skip_index: int | None = None) -> FlowTransformer:
    m, dv = cfg["model"], cfg["derived"]
    return FlowTransformer(
        enc_in=enc_in,
        dec_in=dec_in,
        history_steps=dv["history_steps"],
        n_decoder_tokens=dv["n_decoder_tokens"],
        n_output_tokens=dv["n_output_tokens"],
        token_steps=dv["token_steps"],
        quantiles=m["quantiles"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_encoder_layers=m["n_encoder_layers"],
        n_decoder_layers=m["n_decoder_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m["dropout"],
        encoder_patch_steps=m["encoder_patch_steps"],
        n_catchments=cfg.get("n_catchments", 1),
        geosmart_skip_index=geosmart_skip_index,
    )


def _eligible(samples: D.SampleSet, use_gefs: bool, use_geosmart: bool) -> np.ndarray:
    mask = np.ones(len(samples), dtype=bool)
    if use_gefs:
        mask &= samples.has_gefs
    if use_geosmart:
        mask &= samples.has_geosmart
    return mask


def _fit(model, train_t, val_t, cfg, device, members=None) -> Dict:
    """`members` = (scaled GeoSmart members (N, M, T), member counts (N,), decoder channel): each time a sample is
    used, one of its members is drawn at random and written into that channel."""
    t = cfg["training"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    extra = () if members is None else (torch.from_numpy(members[0]), torch.from_numpy(members[1]))
    loader = DataLoader(TensorDataset(*train_t, *extra), batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=t["batch_size"])
    q = model.quantiles.to(device)

    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = 0.0
        for enc, dec, y, *mem in loader:
            if mem:
                g, n = mem
                pick = (torch.rand(len(n)) * n).long()
                dec = dec.clone()
                dec[:, :, members[2]] = g[torch.arange(len(n)), pick]
            enc, dec, y = enc.to(device), dec.to(device), y.to(device)
            opt.zero_grad()
            loss = pinball_loss(model(enc, dec), y, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_t[0])

        model.eval()
        vt = 0.0
        with torch.no_grad():
            for enc, dec, y in val_loader:
                enc, dec, y = enc.to(device), dec.to(device), y.to(device)
                vt += pinball_loss(model(enc, dec), y, q).item() * len(y)
        val_loss = vt / len(val_t[0])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("epoch %3d  train %.4f  val %.4f", epoch, train_loss, val_loss)

        if val_loss < best - 1e-6:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                logger.info("Early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best}


def _add_skill(model_m: dict, base_m: dict, baseline: str) -> None:
    """skill = 1 - model / baseline quantile loss for a regime and each of its weeks (None where there are no steps)."""
    for m, b in [(model_m, base_m)] + [(model_m["by_week"][w], base_m["by_week"][w]) for w in model_m["by_week"]]:
        m[f"skill_vs_{baseline}"] = 1.0 - m["quantile_loss"] / b["quantile_loss"] if m["n_steps"] else None


def train_variant(cfg: dict, hde: pd.DataFrame, samples: D.SampleSet, variant: dict, mask: np.ndarray, device) -> dict:
    d, dv = cfg["data"], cfg["derived"]
    sub = samples.subset(np.flatnonzero(mask))
    gap = pd.Timedelta(minutes=d["step_minutes"]) * (dv["history_steps"] + dv["horizon_fine_steps"])
    split_idx = D.purged_chronological_split(
        sub.origins, gap, d["splits"]["train_fraction"], d["splits"]["val_fraction"]
    )
    need = d["splits"]["min_samples_per_split"]
    sizes = {k: len(v) for k, v in split_idx.items()}
    if min(sizes.values()) < need:
        raise RuntimeError(
            f"Variant '{variant['name']}': only {len(sub)} usable daily samples -> split sizes {sizes} after "
            f"purging a {gap.days}-day gap between splits (need >= {need} each). Add more data (e.g. more GeoSmart "
            f"files / GEFS history) or shorten data.history_days / data.horizon_days."
        )
    splits = {k: D.geosmart_representation(
        sub.subset(v), variant.get("geosmart_mode", "level"), variant.get("geosmart_ensemble", "members"))
        for k, v in split_idx.items()}
    D.assert_no_temporal_overlap(sub, split_idx)

    quantiles = cfg["model"]["quantiles"]
    mode = d["target_mode"]
    # seasonal climatology: training-period flow only (never looks past the end of the train windows)
    table = seasonal_table(hde, splits["train"].window_end.max(), d["output_step_minutes"], quantiles,
                           d["seasonal_window_days"])
    offsets = {k: target_offset(mode, s, table, quantiles, d["output_step_minutes"]) for k, s in splits.items()}

    scalers = D.Scalers.fit(splits["train"], offsets["train"])
    channels = D.variant_dec_channels(variant["use_gefs"], variant["use_geosmart"], mode in ANCHOR_MODES)
    tensors = {k: tuple(torch.from_numpy(a) for a in scalers.transform(s, channels, offsets[k]))
               for k, s in splits.items()}

    threshold = D.historic_flow_threshold(hde, splits["train"].window_end.max(), d["extreme"]["quantile"])

    skip = channels.index(D.GEO_CH) if variant["use_geosmart"] and variant.get("geosmart_skip") else None
    model = build_model(cfg, len(D.ENC_CHANNELS), len(channels), skip).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[%s] samples %s, decoder channels %s, %d parameters, extreme threshold %.3f m3/s",
                variant["name"], sizes, [D.DEC_CHANNELS[c] for c in channels], n_params, threshold)

    ens = cfg["ensemble"]
    geo_pos = channels.index(D.GEO_CH) if variant["use_geosmart"] else None
    members = {k: scalers.transform_members(s) for k, s in splits.items()} if geo_pos is not None else {}
    train_members, val_t = None, tensors["val"]
    if geo_pos is not None:
        train_members = (members["train"], splits["train"].geo_n, geo_pos)
        val_t = expand_validation(
            tensors["val"], members["val"], splits["val"].geo_n, geo_pos, ens["val_members_per_sample"],
            cfg["training"]["seed"])
    fit = _fit(model, tensors["train"], val_t, cfg, device, train_members)

    metrics = {}
    for name in ("val", "test"):
        enc, dec, _ = (t.numpy() for t in tensors[name])
        pred = predict_flow(
            model, scalers, enc, dec, device, quantiles, offsets[name],
            members.get(name), splits[name].geo_n, geo_pos, ens["pool_draws_per_member"])
        args = (quantiles, dv["steps_per_day"], threshold, d["extreme"]["prediction_quantile"])
        y, n = splits[name].y, len(splits[name])
        metrics[name] = evaluate_split(y, pred, *args)
        base_pred = {
            "climatology": climatology(splits["train"].y, n, quantiles),
            "seasonal_climatology": seasonal_climatology(
                table, splits[name].origins, dv["horizon_steps"], d["output_step_minutes"]),
            "persistence": persistence(splits[name].last_flow, dv["horizon_steps"], len(quantiles)),
        }
        metrics[name]["baselines"] = {b: evaluate_split(y, p, *args) for b, p in base_pred.items()}
        for b, p in base_pred.items():
            metrics[name]["baselines"][b]["_pred"] = p
        # skill: 1 = perfect, 0 = no better than the baseline; overall and for every forecast week
        for b, bm in metrics[name]["baselines"].items():
            metrics[name][f"skill_vs_{b}"] = 1.0 - metrics[name]["quantile_loss"] / bm["quantile_loss"]
            for wk, wm in metrics[name]["by_week"].items():
                wm[f"skill_vs_{b}"] = 1.0 - wm["quantile_loss"] / bm["by_week"][wk]["quantile_loss"]

        # the same metrics for baseflow- vs runoff-dominated target steps (Boughton filter on observed flow)
        bf = d["baseflow"]
        regime_args = (quantiles, dv["steps_per_day"], splits[name].bfi, bf["bfi_threshold"])
        metrics[name]["by_regime"] = regime_metrics(y, pred, *regime_args)
        for b, bm in metrics[name]["baselines"].items():
            bm["by_regime"] = regime_metrics(y, bm["_pred"], *regime_args)
            for regime, rm in metrics[name]["by_regime"].items():
                _add_skill(rm, bm["by_regime"][regime], b)
        for bm in metrics[name]["baselines"].values():
            bm.pop("_pred")
    return {
        "variant": variant, "model": model, "scalers": scalers, "channels": channels, "threshold": threshold,
        "target_mode": mode,
        "fit": fit, "metrics": metrics, "split_sizes": sizes, "n_parameters": n_params,
        "mean_members": float(splits["test"].geo_n.mean()),
        "split_ranges": {k: [str(s.origins.min()), str(s.origins.max())] for k, s in splits.items()},
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def _save_variant(cfg: dict, res: dict, run_id: str, models_root: Path, results_root: Path) -> dict:
    name = res["variant"]["name"]
    run_cfg = snapshot(cfg)
    run_cfg["variant"] = res["variant"]
    run_cfg["run_id"] = run_id

    mdir = models_root / name
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": res["model"].state_dict(),
            "config": run_cfg,
            "scalers": res["scalers"].to_dict(),
            "decoder_channels": [D.DEC_CHANNELS[c] for c in res["channels"]],
            "extreme_threshold_m3s": res["threshold"],
        },
        mdir / "model.pt",
    )
    with open(mdir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)
    pd.DataFrame(res["fit"]["history"]).to_csv(mdir / "training_history.csv", index=False)

    record = {
        "run_id": run_id,
        "variant": res["variant"],
        "config": run_cfg,
        "n_parameters": res["n_parameters"],
        "split_sizes": res["split_sizes"],
        "split_origin_ranges": res["split_ranges"],
        "best_epoch": res["fit"]["best_epoch"],
        "metrics": res["metrics"],
        "model_dir": str(mdir),
    }
    with open(results_root / f"metrics_{name}.json", "w") as f:
        json.dump(record, f, indent=2)

    row = {"run_id": run_id, "variant": name, **{f"variant.{k}": v for k, v in res["variant"].items() if k != "name"}}
    row.update(_flatten({"data": cfg["data"], "model": cfg["model"], "training": cfg["training"]}))
    row.update({"n_parameters": res["n_parameters"], "best_epoch": res["fit"]["best_epoch"],
                **{f"n_{k}": v for k, v in res["split_sizes"].items()}})
    for split in ("val", "test"):
        m = res["metrics"][split]
        row[f"{split}.quantile_loss"] = m["quantile_loss"]
        row[f"{split}.median_mae"] = m["median_mae"]
        for b, bm in m["baselines"].items():
            row[f"{split}.baseline.{b}.quantile_loss"] = bm["quantile_loss"]
            row[f"{split}.baseline.{b}.median_mae"] = bm["median_mae"]
            row[f"{split}.skill_vs_{b}"] = m[f"skill_vs_{b}"]
        row[f"{split}.extreme.threshold_m3s"] = m["extreme_events"]["threshold_m3s"]
        for regime, rm in m["by_regime"].items():
            for scope, sm in {"": rm, **{f"{wk}.": wm for wk, wm in rm["by_week"].items()}}.items():
                for k in ("n_steps", "quantile_loss", "median_mae", *(f"skill_vs_{b}" for b in m["baselines"])):
                    row[f"{split}.regime.{regime}.{scope}{k}"] = sm.get(k)
        for scope, sm in {"": m, **{f"{wk}.": wm for wk, wm in m["by_week"].items()}}.items():
            if scope:
                for k in ("quantile_loss", "median_mae", *(f"skill_vs_{b}" for b in m["baselines"])):
                    row[f"{split}.{scope}{k}"] = sm[k]
            for level in ("per_step", "per_window"):
                for k in ("false_negative_rate", "false_positive_rate", "precision", "accuracy", "f1"):
                    row[f"{split}.{scope}extreme.{level}.{k}"] = sm["extreme_events"][level][k]
    return row


def _comparison(rows: List[dict]) -> pd.DataFrame:
    keep = ["variant", "n_train", "n_val", "n_test", "test.quantile_loss", "test.median_mae",
            "test.baseline.climatology.quantile_loss", "test.baseline.seasonal_climatology.quantile_loss",
            "test.baseline.persistence.quantile_loss", "test.skill_vs_climatology",
            "test.skill_vs_seasonal_climatology", "test.skill_vs_persistence"] + [
        c for c in rows[0] if c.startswith("test.extreme.") or c.startswith("test.week_") or c.startswith("test.regime.")
    ]
    df = pd.DataFrame(rows)[keep].set_index("variant")
    if len(df) == 2:
        a, b = df.index
        diff = (df.loc[a].apply(pd.to_numeric, errors="coerce") - df.loc[b].apply(pd.to_numeric, errors="coerce"))
        df.loc[f"{a} - {b}"] = diff
    return df


def run_training(cfg: dict, hde: pd.DataFrame, gefs, geosmart, run_id: Optional[str] = None) -> pd.DataFrame:
    t = cfg["training"]
    torch.manual_seed(t["seed"])
    np.random.seed(t["seed"])
    device = pick_device(t["device"])
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    bh = cfg["borehole_key"]
    models_root = Path(cfg["paths"]["models_dir"]) / bh / run_id
    results_root = Path(cfg["paths"]["results_dir"]) / bh / run_id
    results_root.mkdir(parents=True, exist_ok=True)

    variants = cfg["ablation"]["variants"]
    samples = D.build_samples(
        cfg, hde,
        gefs if any(v["use_gefs"] for v in variants) else None,
        geosmart if any(v["use_geosmart"] for v in variants) else None,
    )

    common = np.ones(len(samples), dtype=bool)
    for v in variants:
        common &= _eligible(samples, v["use_gefs"], v["use_geosmart"])
    logger.info("Device %s | %d samples, %d usable by every variant", device, len(samples), common.sum())

    rows = []
    for v in variants:
        mask = common if cfg["ablation"]["restrict_to_common_samples"] else _eligible(samples, v["use_gefs"], v["use_geosmart"])
        torch.manual_seed(t["seed"])
        start = time.time()
        res = train_variant(cfg, hde, samples, v, mask, device)
        logger.info("[%s] finished in %.0fs", v["name"], time.time() - start)
        rows.append(_save_variant(cfg, res, run_id, models_root, results_root))

    comparison = _comparison(rows)
    comparison.to_csv(results_root / "comparison.csv")
    with open(results_root / "config.yaml", "w") as f:
        yaml.safe_dump(snapshot(cfg), f, sort_keys=False)

    summary_path = results_root.parent / "summary.csv"
    new = pd.DataFrame(rows)
    if summary_path.exists():
        new = pd.concat([pd.read_csv(summary_path), new], ignore_index=True)
    new.to_csv(summary_path, index=False)

    logger.info("Results -> %s | weights -> %s", results_root, models_root)
    return comparison


def load_checkpoint(path) -> tuple:
    """Rebuild a trained model from models/<borehole>/<run_id>/<variant>/model.pt."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    skip = None
    if cfg.get("variant", {}).get("geosmart_skip"):
        skip = ckpt["decoder_channels"].index(D.DEC_CHANNELS[D.GEO_CH])
    model = build_model(cfg, len(D.ENC_CHANNELS), len(ckpt["decoder_channels"]), skip)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
