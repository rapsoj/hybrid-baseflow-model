"""Multi-seed final comparison of the joint chalk-catchment model with vs without GeoSmart forecasts."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .multi import run_multi_catchment
from .palette import BLACK, PURPLE, tint

logger = logging.getLogger(__name__)

SCOPES = ("overall", "baseflow_dominated", "runoff_dominated")
SCOPE_TITLES = {"overall": "Overall", "baseflow_dominated": "Baseflow-dominated", "runoff_dominated": "Runoff-dominated"}
SCALARS = ("quantile_loss", "median_mae", "coverage_0.05_0.95", "nominal_coverage", "skill_vs_climatology",
           "skill_vs_seasonal_climatology", "skill_vs_persistence", "n_samples", "n_steps", "share_of_steps")


def _scalars(block: dict) -> dict:
    out = {k: block[k] for k in SCALARS if k in block and not isinstance(block[k], (dict, list))}
    by_q = block.get("quantile_loss_by_quantile")
    if isinstance(by_q, dict):
        out.update({f"quantile_loss_q{q}": v for q, v in by_q.items()})
    return out


def _rows(seed: int, metrics: dict) -> tuple[list, list]:
    """Flatten one run's metrics.json into long rows (every variant, catchment, split, regime and forecast week)."""
    rows, extremes = [], []
    for variant, per_catchment in metrics.items():
        for catchment, splits in per_catchment.items():
            for split, m in splits.items():
                key = dict(seed=seed, variant=variant, catchment=catchment, split=split)
                scopes = {"overall": m, **m.get("by_regime", {})}
                for scope, block in scopes.items():
                    rows.append({**key, "scope": scope, "week": "all", **_scalars(block)})
                    for week, wblock in block.get("by_week", {}).items():
                        rows.append({**key, "scope": scope, "week": week, **_scalars(wblock)})
                for baseline, b in m.get("baselines", {}).items():
                    bkey = {**key, "variant": f"reference:{baseline}"}
                    rows.append({**bkey, "scope": "overall", "week": "all", **_scalars(b)})
                    for week, wblock in b.get("by_week", {}).items():
                        rows.append({**bkey, "scope": "overall", "week": week, **_scalars(wblock)})
                    for regime, rblock in b.get("by_regime", {}).items():
                        rows.append({**bkey, "scope": regime, "week": "all", **_scalars(rblock)})
                        for week, wblock in rblock.get("by_week", {}).items():
                            rows.append({**bkey, "scope": regime, "week": week, **_scalars(wblock)})
                ev = m.get("extreme_events", {})
                flat = {f"{level}.{k}": v for level in ("per_step", "per_window") for k, v in ev.get(level, {}).items()}
                extremes.append({**key, "threshold_m3s": ev.get("threshold_m3s"), **flat})
    return rows, extremes


def _paired(long: pd.DataFrame, baseline: str, geosmart: str, split: str = "test") -> pd.DataFrame:
    """Rows keyed by (seed, catchment, scope, week) with baseline and GeoSmart values side by side."""
    sub = long[(long["split"] == split) & long["variant"].isin([baseline, geosmart])]
    keys = ["seed", "catchment", "scope", "week"]
    cols = ["quantile_loss", "median_mae", "skill_vs_persistence"]
    wide = sub.pivot_table(index=keys, columns="variant", values=cols, aggfunc="first")
    wide.columns = [f"{c}.{v}" for c, v in wide.columns]
    return wide.reset_index()


def _seed_curves(paired: pd.DataFrame, baseline: str, geosmart: str, scope: str, metric: str, relative: bool):
    """(weeks, values[variant] -> array (n_seeds, n_weeks)): catchment-mean per seed, then spread across seeds."""
    weeks = sorted(w for w in paired["week"].unique() if w != "all")
    sub = paired[paired["scope"] == scope]
    out = {}
    for variant in (baseline, geosmart):
        per_seed = []
        for _, g in sub.groupby("seed"):
            row = []
            for w in weeks:
                gw = g[g["week"] == w]
                v = gw[f"{metric}.{variant}"].to_numpy(dtype=float)
                if relative:
                    b = gw[f"{metric}.{baseline}"].to_numpy(dtype=float)
                    with np.errstate(all="ignore"):
                        v = np.where(b > 0, v / b, np.nan)
                row.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
            per_seed.append(row)
        out[variant] = np.array(per_seed)
    return weeks, out


def plot_final_by_horizon(long: pd.DataFrame, baseline: str, geosmart: str, out_path: Path) -> Path:
    """
    Rows: overall / baseflow-dominated / runoff-dominated target steps. Columns: quantile loss and median MAE relative
    to the baseline (1 = baseline, below 1 = better) and skill vs persistence. Lines are catchment means; bands show
    the min-max across seeds.
    """
    import matplotlib.pyplot as plt

    paired = _paired(long, baseline, geosmart)
    panels = (("quantile_loss", "Quantile loss relative to baseline", True),
              ("median_mae", "Median MAE relative to baseline", True),
              ("skill_vs_persistence", "Skill vs persistence", False))
    n_seeds = paired["seed"].nunique()
    fig, axes = plt.subplots(len(SCOPES), 3, figsize=(18, 4.6 * len(SCOPES)), squeeze=False)
    for r, scope in enumerate(SCOPES):
        for c, (metric, title, relative) in enumerate(panels):
            ax = axes[r, c]
            weeks, curves = _seed_curves(paired, baseline, geosmart, scope, metric, relative)
            x = np.arange(1, len(weeks) + 1)
            base_curve = curves[baseline].mean(axis=0)
            ax.plot(x, base_curve, color=BLACK, linewidth=3.5, zorder=1,
                    label=f"Baseline: {baseline}" + (" (= 1.0)" if relative else ""))
            if not relative:
                ax.fill_between(x, curves[baseline].min(0), curves[baseline].max(0), color=BLACK, alpha=0.12)
            g = curves[geosmart]
            ax.fill_between(x, np.nanmin(g, axis=0), np.nanmax(g, axis=0), color=PURPLE, alpha=0.22, zorder=2)
            ax.plot(x, np.nanmean(g, axis=0), marker="o", color=PURPLE, linewidth=2.2, zorder=3,
                    label=f"{geosmart} (mean, band = seed min-max)")
            ax.set_xticks(x)
            ax.set_xticklabels([w.replace("week_", "") for w in weeks])
            ax.set_xlabel("Forecast week")
            ax.set_title(f"{SCOPE_TITLES[scope]}: {title}")
            ax.grid(alpha=0.25)
            if not relative:
                ax.axhline(0, color=BLACK, linewidth=0.8)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Joint chalk-catchment model, test set: with vs without GeoSmart by forecast week "
        f"({n_seeds} seeds, catchment mean)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_final_by_catchment(long: pd.DataFrame, baseline: str, geosmart: str, out_path: Path) -> Path:
    """Per-catchment overall test quantile loss change vs the baseline: seed mean +/- std and every seed as a dot."""
    import matplotlib.pyplot as plt

    paired = _paired(long, baseline, geosmart)
    p = paired[(paired["scope"] == "overall") & (paired["week"] == "all")].copy()
    p["change_pct"] = 100 * (p[f"quantile_loss.{geosmart}"] / p[f"quantile_loss.{baseline}"] - 1)
    catchments = list(dict.fromkeys(p["catchment"]))
    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.55 * (len(catchments) + 1)))
    labels = catchments + ["MEAN"]
    for i, c in enumerate(labels):
        if c == "MEAN":
            vals = p.groupby("seed")["change_pct"].mean().to_numpy()
        else:
            vals = p[p["catchment"] == c]["change_pct"].to_numpy()
        color = PURPLE if vals.mean() < 0 else tint(BLACK, 0.55)
        ax.barh(i, vals.mean(), xerr=vals.std(ddof=0), color=color, alpha=0.75, capsize=3,
                hatch="//" if c == "MEAN" else None)
        ax.scatter(vals, np.full(len(vals), i), color=BLACK, s=12, zorder=3)
    ax.axvline(0, color=BLACK, linewidth=3.0, label=f"Baseline: {baseline}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in test quantile loss vs baseline (%, negative = GeoSmart better); bars mean ± std, dots seeds")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"{geosmart} vs {baseline} by catchment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarise(long: pd.DataFrame, baseline: str, geosmart: str) -> pd.DataFrame:
    """Per catchment (and the catchment mean): overall test metrics for both variants, change and how many seeds win."""
    paired = _paired(long, baseline, geosmart)
    p = paired[(paired["scope"] == "overall") & (paired["week"] == "all")].copy()
    p["rel_ql"] = p[f"quantile_loss.{geosmart}"] / p[f"quantile_loss.{baseline}"]
    p["rel_mae"] = p[f"median_mae.{geosmart}"] / p[f"median_mae.{baseline}"]
    rows = []
    groups = [(c, g) for c, g in p.groupby("catchment", sort=False)]
    mean_g = p.groupby("seed")[["rel_ql", "rel_mae", f"quantile_loss.{baseline}", f"quantile_loss.{geosmart}",
                                f"median_mae.{baseline}", f"median_mae.{geosmart}"]].mean().reset_index()
    for name, g in groups + [("MEAN (catchments)", mean_g)]:
        rows.append({
            "catchment": name,
            "n_seeds": len(g),
            f"quantile_loss_{baseline}": g[f"quantile_loss.{baseline}"].mean(),
            f"quantile_loss_{geosmart}": g[f"quantile_loss.{geosmart}"].mean(),
            "quantile_loss_geosmart_std": g[f"quantile_loss.{geosmart}"].std(ddof=0),
            "rel_quantile_loss": g["rel_ql"].mean(),
            "rel_quantile_loss_std": g["rel_ql"].std(ddof=0),
            "rel_median_mae": g["rel_mae"].mean(),
            "seeds_geosmart_better": int((g["rel_ql"] < 1).sum()),
        })
    return pd.DataFrame(rows)


def run_final(config_path="config.yaml", overrides=(), run_id: str | None = None) -> pd.DataFrame:
    """Train baseline and GeoSmart variants for every configured seed, then aggregate metrics and plots."""
    base = load_config(config_path, "chilgrove", overrides)
    final = base["final_run"]
    seeds, baseline, geosmart = final["seeds"], final["baseline"], final["geosmart"]
    run_id = run_id or "final-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    results_root = Path(base["paths"]["results_dir"]) / "all_chalk" / run_id
    results_root.mkdir(parents=True, exist_ok=True)

    long_rows, extreme_rows, split_sizes, catchments = [], [], {}, []
    for seed in seeds:
        seed_id = f"{run_id}/seed_{seed}"
        metrics_path = Path(base["paths"]["results_dir"]) / "all_chalk" / seed_id / "metrics.json"
        if metrics_path.is_file():
            logger.info("Seed %s: reusing finished run %s", seed, seed_id)
        else:
            logger.info("Seed %s: training %s and %s", seed, baseline, geosmart)
            run_multi_catchment(config_path, list(overrides) + [f"training.seed={seed}"], seed_id, [baseline, geosmart])
        with open(metrics_path) as f:
            saved = json.load(f)
        rows, extremes = _rows(seed, saved["metrics"])
        long_rows += rows
        extreme_rows += extremes
        split_sizes, catchments = saved["split_sizes"], saved["catchment_keys"]

    long = pd.DataFrame(long_rows)
    long.to_csv(results_root / "metrics_all.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(results_root / "extreme_event_metrics.csv", index=False)
    summary = summarise(long, baseline, geosmart)
    summary.to_csv(results_root / "summary_by_catchment.csv", index=False)
    plots = [
        plot_final_by_horizon(long, baseline, geosmart, results_root / "metrics_by_horizon.png"),
        plot_final_by_catchment(long, baseline, geosmart, results_root / "metrics_by_catchment.png"),
    ]
    with open(results_root / "final_config.json", "w") as f:
        json.dump({"seeds": seeds, "baseline": baseline, "geosmart": geosmart, "catchments": catchments,
                   "split_sizes": split_sizes}, f, indent=2)
    logger.info("Final results -> %s | plots %s", results_root, [str(p) for p in plots])
    return summary
