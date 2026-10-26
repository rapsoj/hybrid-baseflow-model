"""Reusable visualisations for run comparisons and sample forecasts.

Example:
    .venv/bin/python -m pipeline.visualize chilgrove --run-id wk4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import data as D
from .config import load_config
from .palette import BLACK, BLUE, PURPLE
from .ensemble import predict_flow, predict_member_flow
from .target import ANCHOR_MODES, seasonal_table, target_offset
from .train import load_checkpoint, pick_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("borehole", help="borehole key from config.yaml (e.g. chilgrove)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--run-id", default=None, help="results run id (default: latest run with both variants)")
    p.add_argument("--n-random", type=int, default=6, help="number of random test origins to plot")
    p.add_argument("--seed", type=int, default=42, help="random seed for sample selection")
    p.add_argument("--device", default="auto", help="inference device: auto|cpu|cuda|mps")
    return p.parse_args(argv)


def _latest_run_id(results_root: Path) -> str:
    runs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and (p / "metrics_with_geosmart.json").exists()
         and (p / "metrics_without_geosmart.json").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"No complete runs found under {results_root}")
    return runs[-1].name


def _load_metrics_pair(run_dir: Path) -> Dict[str, dict]:
    out = {}
    for name in ("with_geosmart", "without_geosmart"):
        path = run_dir / f"metrics_{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        out[name] = json.loads(path.read_text())
    return out


def _merge_nested(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_nested(out[k], v)
        else:
            out[k] = v
    return out


def _eligible(samples: D.SampleSet, use_gefs: bool, use_geosmart: bool) -> np.ndarray:
    mask = np.ones(len(samples), dtype=bool)
    if use_gefs:
        mask &= samples.has_gefs
    if use_geosmart:
        mask &= samples.has_geosmart
    return mask


def _split_for_variant(cfg: dict, samples: D.SampleSet, variant: dict, common: np.ndarray) -> Dict[str, D.SampleSet]:
    d, dv = cfg["data"], cfg["derived"]
    mask = common if cfg["ablation"]["restrict_to_common_samples"] else _eligible(
        samples, variant["use_gefs"], variant["use_geosmart"]
    )
    sub = samples.subset(np.flatnonzero(mask))
    gap = pd.Timedelta(minutes=d["step_minutes"]) * (dv["history_steps"] + dv["horizon_fine_steps"])
    idx = D.purged_chronological_split(sub.origins, gap, d["splits"]["train_fraction"], d["splits"]["val_fraction"])
    return {k: sub.subset(v) for k, v in idx.items()}


def _scalers_from_ckpt(raw: dict) -> D.Scalers:
    return D.Scalers(
        enc_mean=np.asarray(raw["enc_mean"], dtype=np.float32),
        enc_std=np.asarray(raw["enc_std"], dtype=np.float32),
        dec_mean=np.asarray(raw["dec_mean"], dtype=np.float32),
        dec_std=np.asarray(raw["dec_std"], dtype=np.float32),
        y_mean=float(raw["y_mean"]),
        y_std=float(raw["y_std"]),
    )


def _predict_variant(
    cfg: dict,
    variant: dict,
    ckpt_path: Path,
    splits: Dict[str, D.SampleSet],
    hde: pd.DataFrame,
    device: torch.device,
) -> dict:
    model, ckpt = load_checkpoint(ckpt_path)
    model = model.to(device)
    quantiles = cfg["model"]["quantiles"]
    mode = cfg["data"]["target_mode"]
    dv, d = cfg["derived"], cfg["data"]
    train_end = splits["train"].window_end.max()
    table = seasonal_table(hde, train_end, d["output_step_minutes"], quantiles, d["seasonal_window_days"])
    offsets = {k: target_offset(mode, s, table, quantiles, d["output_step_minutes"]) for k, s in splits.items()}
    scalers = _scalers_from_ckpt(ckpt["scalers"])

    unknown = [name for name in ckpt["decoder_channels"] if name not in D.DEC_CHANNELS]
    if unknown:
        raise RuntimeError(
            "This run uses legacy decoder channels that are not supported by the current plotting pipeline: "
            f"{unknown}. Re-run training with the current configuration or extend the plotting adapter."
        )
    channels = [D.DEC_CHANNELS.index(name) for name in ckpt["decoder_channels"]]
    test = splits["test"]
    enc, dec, _ = scalers.transform(test, channels, offsets["test"])

    out = {
        "origins": test.origins,
        "truth": test.y,
        "pred": None,
        "member_pred": None,
        "quantiles": np.asarray(quantiles, dtype=float),
        "horizon_steps": dv["horizon_steps"],
        "step_hours": d["output_step_minutes"] / 60.0,
    }
    if variant["use_geosmart"]:
        geo_pos = channels.index(D.GEO_CH)
        members = scalers.transform_members(test)
        out["pred"] = predict_flow(
            model, scalers, enc, dec, device, quantiles, offsets["test"],
            members, test.geo_n, geo_pos, cfg["ensemble"]["pool_draws_per_member"],
        )
        out["member_pred"] = predict_member_flow(model, scalers, enc, dec, device, offsets["test"], members, test.geo_n, geo_pos)
        out["n_members"] = test.geo_n.copy()
    else:
        out["pred"] = predict_flow(model, scalers, enc, dec, device, quantiles, offsets["test"])
        out["n_members"] = np.zeros(len(test), dtype=int)
    return out


def _build_model_summaries(metrics_pair: Dict[str, dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for variant, rec in metrics_pair.items():
        m = rec["metrics"]["test"]
        row = {
            "variant": variant,
            "overall.quantile_loss": m["quantile_loss"],
            "overall.median_mae": m["median_mae"],
            "overall.skill_vs_persistence": m["skill_vs_persistence"],
        }
        for wk, wm in m["by_week"].items():
            row[f"{wk}.quantile_loss"] = wm["quantile_loss"]
            row[f"{wk}.median_mae"] = wm["median_mae"]
            row[f"{wk}.skill_vs_persistence"] = wm["skill_vs_persistence"]
        for regime, rm in m["by_regime"].items():
            row[f"regime.{regime}.quantile_loss"] = rm.get("quantile_loss")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def _plot_random_predictions(out_dir: Path, with_geo: dict, without_geo: dict, n_random: int, seed: int) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install dependencies from requirements.txt.") from exc

    common = with_geo["origins"].intersection(without_geo["origins"])
    if len(common) == 0:
        raise RuntimeError("No overlapping test origins between variants")
    rng = np.random.default_rng(seed)
    chosen = common[rng.choice(len(common), size=min(n_random, len(common)), replace=False)]
    chosen = pd.DatetimeIndex(sorted(chosen))
    pd.Series(chosen).to_csv(out_dir / "random_prediction_origins.csv", index=False, header=["origin_utc"])

    q = with_geo["quantiles"]
    q50 = int(np.argmin(np.abs(q - 0.5)))
    qlo, qhi = 0, len(q) - 1
    x = (np.arange(with_geo["horizon_steps"]) + 1) * with_geo["step_hours"] / 24.0
    n_cols = 3
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.8 * n_rows), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    idx_with = {t: i for i, t in enumerate(with_geo["origins"])}
    idx_without = {t: i for i, t in enumerate(without_geo["origins"])}
    for i, t0 in enumerate(chosen):
        ax = axes[i]
        iw, ino = idx_with[t0], idx_without[t0]
        y = with_geo["truth"][iw]
        pw = with_geo["pred"][iw]
        pno = without_geo["pred"][ino]

        member_median = None
        member_min = member_max = None
        members = with_geo["member_pred"][iw]
        if members is not None and len(members):
            member_median = members[:, :, q50]
            member_min = member_median.min(axis=0)
            member_max = member_median.max(axis=0)
            for mline in member_median:
                ax.plot(x, mline, color=PURPLE, alpha=0.12, lw=0.7)
            ax.fill_between(x, member_min, member_max, color=PURPLE, alpha=0.16, label="GeoSmart member range")

        ax.plot(x, y, color=BLACK, lw=1.8, label="Observed (target)")
        ax.plot(x, pw[:, q50], color=PURPLE, lw=2.0, label="With GeoSmart (pooled median)")
        ax.fill_between(x, pw[:, qlo], pw[:, qhi], color=PURPLE, alpha=0.22, label="With GeoSmart P05-P95")
        ax.plot(x, pno[:, q50], color=BLUE, lw=2.0, label="Without GeoSmart (median)")
        ax.fill_between(x, pno[:, qlo], pno[:, qhi], color=BLUE, alpha=0.20, label="Without GeoSmart P05-P95")
        ax.set_title(f"Origin {t0:%Y-%m-%d %H:%M} UTC | GeoSmart members {int(with_geo['n_members'][iw])}")
        ax.set_ylabel("Target flow (m³/s)")
        ax.grid(alpha=0.25, linewidth=0.6)

    for j in range(len(chosen), len(axes)):
        axes[j].axis("off")
    for ax in axes[max(0, len(axes) - n_cols):]:
        ax.set_xlabel("Lead time (days)")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Random test forecasts: with vs without GeoSmart", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "predictions_random6.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_accuracy_summary(out_dir: Path, metrics_pair: Dict[str, dict]) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install dependencies from requirements.txt.") from exc

    df = _build_model_summaries(metrics_pair)
    variants = list(df.index)
    labels = ["overall", "week_1", "week_2", "week_3", "week_4"]
    x = np.arange(len(labels))
    w = 0.36
    colors = [PURPLE, BLACK]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    def bars(ax, metric_key: str, title: str, ylabel: str):
        for i, v in enumerate(variants):
            vals = [df.loc[v, f"{labels[0]}.{metric_key}"]] + [df.loc[v, f"{wk}.{metric_key}"] for wk in labels[1:]]
            ax.bar(x + (i - 0.5) * w, vals, width=w, color=colors[i % len(colors)], alpha=0.88, label=v)
        ax.set_xticks(x)
        ax.set_xticklabels(["Overall", "W1", "W2", "W3", "W4"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    bars(axs[0, 0], "quantile_loss", "Quantile loss by horizon", "m³/s")
    bars(axs[0, 1], "median_mae", "Median MAE by horizon", "m³/s")
    bars(axs[1, 0], "skill_vs_persistence", "Skill vs persistence by horizon", "1 - model_loss / persistence_loss")

    reg_labels = ["baseflow_dominated", "runoff_dominated"]
    xr = np.arange(len(reg_labels))
    for i, v in enumerate(variants):
        vals = [df.loc[v, f"regime.{r}.quantile_loss"] for r in reg_labels]
        axs[1, 1].bar(xr + (i - 0.5) * w, vals, width=w, color=colors[i % len(colors)], alpha=0.88, label=v)
    axs[1, 1].set_xticks(xr)
    axs[1, 1].set_xticklabels(["Baseflow-dominated", "Runoff-dominated"])
    axs[1, 1].set_title("Regime quantile loss")
    axs[1, 1].set_ylabel("m³/s")
    axs[1, 1].grid(axis="y", alpha=0.25, linewidth=0.6)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Model accuracy summary: with vs without GeoSmart", y=1.02, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "accuracy_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    base_cfg = load_config(args.config, args.borehole)
    borehole_results = Path(base_cfg["paths"]["results_dir"]) / args.borehole
    run_id = args.run_id or _latest_run_id(borehole_results)
    run_dir = borehole_results / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run {run_id} not found under {borehole_results}")
    metrics_pair = _load_metrics_pair(run_dir)

    run_cfg = dict(base_cfg)
    stored = metrics_pair["with_geosmart"]["config"]
    for k in ("borehole_key", "borehole", "derived"):
        if k in stored:
            run_cfg[k] = stored[k]
    for k in ("data", "model", "training", "ablation"):
        if k in stored:
            run_cfg[k] = _merge_nested(base_cfg[k], stored[k])

    hde = D.load_hde(run_cfg["paths"]["hde_dir"], run_cfg["data"]["step_minutes"], run_cfg["data"])
    gefs = D.load_gefs(run_cfg["paths"]["gefs_csv"], run_cfg["data"]["gefs"]["bin_hours"])
    geosmart = D.load_geosmart(run_cfg["paths"]["geosmart_dir"], run_cfg["borehole"]["geosmart_prefix"], run_cfg["data"]["geosmart"])
    samples = D.build_samples(run_cfg, hde, gefs, geosmart)
    common = np.ones(len(samples), dtype=bool)
    for v in run_cfg["ablation"]["variants"]:
        common &= _eligible(samples, v["use_gefs"], v["use_geosmart"])

    variants = {v["name"]: v for v in run_cfg["ablation"]["variants"]}
    with_split = _split_for_variant(run_cfg, samples, variants["with_geosmart"], common)
    without_split = _split_for_variant(run_cfg, samples, variants["without_geosmart"], common)

    models_root = Path(run_cfg["paths"]["models_dir"]) / args.borehole / run_id
    device = pick_device(args.device)
    with_pred = _predict_variant(
        run_cfg, variants["with_geosmart"], models_root / "with_geosmart" / "model.pt", with_split, hde, device
    )
    without_pred = _predict_variant(
        run_cfg, variants["without_geosmart"], models_root / "without_geosmart" / "model.pt", without_split, hde, device
    )

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = _plot_random_predictions(out_dir, with_pred, without_pred, args.n_random, args.seed)
    summary_path = _plot_accuracy_summary(out_dir, metrics_pair)
    print(f"Saved {pred_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {out_dir / 'random_prediction_origins.csv'}")


if __name__ == "__main__":
    main()
