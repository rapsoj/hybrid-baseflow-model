"""Plots and summaries for the joint multi-catchment run (any number of ablation variants)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .palette import BLACK, BLUE, PURPLE, VARIANT_COLORS

BASELINE = "without_geosmart"
PANELS = (
    ("quantile_loss", "Quantile loss (m³/s)"),
    ("median_mae", "Median MAE (m³/s)"),
    ("skill_vs_persistence", "Skill vs persistence"),
)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot results; install dependencies from requirements.txt") from exc
    return plt


def _colors(variants: list[str]) -> dict:
    others = [v for v in variants if v != BASELINE]
    colors = {v: VARIANT_COLORS[i % len(VARIANT_COLORS)] for i, v in enumerate(others)}
    return {v: (BLACK if v == BASELINE else colors[v]) for v in variants}


def _num(x):
    return np.nan if x is None else x


def _save(fig, out_path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    _plt().close(fig)
    return out_path


def plot_catchment_metrics(frame: pd.DataFrame, out_path: Path) -> Path:
    plt = _plt()
    catchments = list(dict.fromkeys(frame["catchment"]))
    variants = list(dict.fromkeys(frame["variant"]))
    colors = _colors(variants)
    height = 0.8 / len(variants)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.2 * len(variants) * len(catchments) + 2)), sharey=True)
    y = np.arange(len(catchments))
    for ax, metric, title in ((axes[0], "quantile_loss", "Test quantile loss (m³/s)"),
                              (axes[1], "median_mae", "Test median MAE (m³/s)")):
        for i, variant in enumerate(variants):
            rows = frame[frame["variant"] == variant].set_index("catchment")
            values = [rows.loc[c, metric] if c in rows.index else np.nan for c in catchments]
            ax.barh(y + (i - (len(variants) - 1) / 2) * height, values, height, color=colors[variant], label=variant)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(catchments)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Joint chalk-catchment model: test metrics by catchment", y=1.01)
    return _save(fig, out_path)


def plot_by_horizon(all_metrics: dict, out_path: Path) -> Path:
    """
    Catchment-averaged test metrics by forecast week, one line per variant. Quantile loss and MAE are shown relative
    to the baseline variant in the same catchment (1 = baseline; below 1 = better), so catchments of very different
    flow size can be averaged; skill vs persistence is averaged as is.
    """
    plt = _plt()
    variants = list(all_metrics)
    keys = list(all_metrics[variants[0]])
    weeks = list(all_metrics[variants[0]][keys[0]]["test"]["by_week"])
    x = np.arange(1, len(weeks) + 1)
    colors = _colors(variants)
    panels = (("quantile_loss", "Quantile loss relative to " + BASELINE, True),
              ("median_mae", "Median MAE relative to " + BASELINE, True),
              ("skill_vs_persistence", "Skill vs persistence (catchment mean)", False))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title, relative) in zip(axes, panels):
        for variant in variants:
            curve = []
            for w in weeks:
                values = []
                for k in keys:
                    v = _num(all_metrics[variant][k]["test"]["by_week"][w][metric])
                    if relative:
                        b = _num(all_metrics[BASELINE][k]["test"]["by_week"][w][metric])
                        v = v / b if b else np.nan
                    values.append(v)
                curve.append(np.nanmean(values) if not np.all(np.isnan(values)) else np.nan)
            if variant == BASELINE:
                label = f"Baseline: {BASELINE}" + (" (= 1.0)" if relative else "")
                ax.plot(x, curve, color=BLACK, linewidth=3.5, label=label, zorder=1)
            else:
                ax.plot(x, curve, marker="o", color=colors[variant], label=variant, zorder=2)
        ax.set_xticks(x)
        ax.set_xlabel("Forecast week")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if not relative:
            ax.axhline(0, color=BLACK, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Joint chalk-catchment model: test metrics by forecast horizon", y=1.02)
    return _save(fig, out_path)


def plot_by_regime(all_metrics: dict, out_path: Path) -> Path:
    """Test quantile loss, median MAE and skill vs persistence per catchment for each flow regime."""
    plt = _plt()
    variants = list(all_metrics)
    keys = list(all_metrics[variants[0]])
    regimes = ("baseflow_dominated", "runoff_dominated")
    colors = _colors(variants)
    colors[BASELINE] = BLUE
    height = 0.8 / len(variants)
    fig, axes = plt.subplots(len(regimes), 3, figsize=(17, 2 + 0.22 * len(keys) * len(variants) * len(regimes)),
                             sharey=True)
    y = np.arange(len(keys))
    for r, regime in enumerate(regimes):
        for c, (metric, label) in enumerate(PANELS):
            ax = axes[r, c]
            for i, variant in enumerate(variants):
                values = [_num(all_metrics[variant][k]["test"]["by_regime"][regime].get(metric)) for k in keys]
                ax.barh(y + (i - (len(variants) - 1) / 2) * height, values, height, color=colors[variant],
                        edgecolor="white", linewidth=0.5, label=variant)
            ax.set_title(f"{regime.replace('_', ' ')}: {label}")
            ax.grid(axis="x", alpha=0.25, linewidth=0.6)
            if metric.startswith("skill"):
                ax.axvline(0, color=BLACK, linewidth=0.8)
        axes[r, 0].set_yticks(y)
        axes[r, 0].set_yticklabels(keys)
    axes[0, 0].invert_yaxis()
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle("Joint chalk-catchment model: test metrics by flow regime", y=1.0)
    return _save(fig, out_path)


def variant_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per variant: test quantile loss / MAE relative to the baseline variant, per catchment and averaged."""
    base = frame[frame["variant"] == BASELINE].set_index("catchment")
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        group = group.set_index("catchment")
        rel_ql = (group["quantile_loss"] / base["quantile_loss"]).replace([np.inf, -np.inf], np.nan)
        rel_mae = (group["median_mae"] / base["median_mae"]).replace([np.inf, -np.inf], np.nan)
        rows.append({
            "variant": variant,
            "mean_rel_quantile_loss": rel_ql.mean(),
            "median_rel_quantile_loss": rel_ql.median(),
            "mean_rel_median_mae": rel_mae.mean(),
            "catchments_better_than_baseline": int((rel_ql < 1).sum()),
            "n_catchments": int(rel_ql.notna().sum()),
        })
    return pd.DataFrame(rows)


def plot_vs_baseline(frame: pd.DataFrame, out_path: Path) -> Path:
    """% change in test quantile loss of every GeoSmart variant vs the baseline, per catchment (negative = better)."""
    plt = _plt()
    base = frame[frame["variant"] == BASELINE].set_index("catchment")
    variants = [v for v in dict.fromkeys(frame["variant"]) if v != BASELINE]
    catchments = list(base.index)
    colors = _colors(list(dict.fromkeys(frame["variant"])))
    height = 0.8 / len(variants)
    y = np.arange(len(catchments) + 1)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.22 * len(variants) * (len(catchments) + 1)))
    for i, variant in enumerate(variants):
        group = frame[frame["variant"] == variant].set_index("catchment")
        change = [100 * (group.loc[c, "quantile_loss"] / base.loc[c, "quantile_loss"] - 1) for c in catchments]
        change.append(float(np.nanmean(change)))
        ax.barh(y + (i - (len(variants) - 1) / 2) * height, change, height, color=colors[variant], label=variant)
    ax.axvline(0, color=BLACK, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(catchments + ["MEAN"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Change in test quantile loss vs {BASELINE} (%, negative = better)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("GeoSmart options vs no-GeoSmart baseline")
    return _save(fig, out_path)


def plot_random_predictions(test_predictions: dict, quantiles, step_hours: float, out_dir: Path, seed: int,
                            n_random: int = 6, with_variant: str = "with_geosmart") -> list[Path]:
    """
    One figure per catchment with `n_random` random test origins (the same for every variant): observed flow, the
    baseline and `with_variant` median with their widest quantile band, and the GeoSmart per-member median range.
    The chosen origins are written next to each figure.
    """
    plt = _plt()
    out_dir.mkdir(parents=True, exist_ok=True)
    q = np.asarray(quantiles)
    q50, qlo, qhi = int(np.argmin(np.abs(q - 0.5))), 0, len(q) - 1
    band = f"P{round(100 * q[qlo]):02d}-P{round(100 * q[qhi]):02d}"
    paths = []
    for key, geo in test_predictions[with_variant].items():
        base = test_predictions[BASELINE][key]
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(len(geo["origins"]), size=min(n_random, len(geo["origins"])), replace=False))
        pd.Series(geo["origins"][picks]).to_csv(out_dir / f"{key}_random_prediction_origins.csv", index=False,
                                                header=["origin_utc"])
        x = (np.arange(geo["truth"].shape[1]) + 1) * step_hours / 24.0
        n_cols = 3
        n_rows = int(np.ceil(len(picks) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.8 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, picks):
            members = geo["members"][i] if geo["members"] else None
            if members is not None and len(members):
                for line in members:
                    ax.plot(x, line, color=PURPLE, alpha=0.12, lw=0.7)
                ax.fill_between(x, members.min(0), members.max(0), color=PURPLE, alpha=0.16,
                                label="GeoSmart member range")
            ax.plot(x, geo["truth"][i], color=BLACK, lw=1.8, label="Observed (target)")
            ax.plot(x, geo["pred"][i][:, q50], color=PURPLE, lw=2, label="With GeoSmart (pooled median)")
            ax.fill_between(x, geo["pred"][i][:, qlo], geo["pred"][i][:, qhi], color=PURPLE, alpha=0.22,
                            label=f"With GeoSmart {band}")
            ax.plot(x, base["pred"][i][:, q50], color=BLUE, lw=2, label="Without GeoSmart (median)")
            ax.fill_between(x, base["pred"][i][:, qlo], base["pred"][i][:, qhi], color=BLUE, alpha=0.20,
                            label=f"Without GeoSmart {band}")
            ax.set_title(f"Origin {geo['origins'][i]:%Y-%m-%d %H:%M} UTC | GeoSmart members {int(geo['n_members'][i])}")
            ax.set_ylabel("Target flow (m³/s)")
            ax.grid(alpha=0.25, linewidth=0.6)
        for ax in axes[len(picks):]:
            ax.axis("off")
        for ax in axes[max(0, len(axes) - n_cols):]:
            ax.set_xlabel("Lead time (days)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="upper center",
                   ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"{key}: random test forecasts, with vs without GeoSmart", y=1.02, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"{key}_predictions_random{len(picks)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
