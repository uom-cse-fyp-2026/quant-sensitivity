"""Gate 0 sanity outputs for the pilot: never reported, only used to catch bugs.

- plots/pilot_histograms.png: value histogram of each regime (regenerated from the seeds)
- summary/regime_stats.csv: per-regime medians of the input descriptors
- summary/pilot_medians.csv: median errors per operator x regime x quantizer x policy
- summary/pilot_predictions_check.md: the numbers behind each prediction in reports/predictions.md
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import planned_samples  # noqa: E402
from src.distributions import generate_stratum  # noqa: E402
from src.metrics import PRIMARY  # noqa: E402

# Reference palette (dataviz skill), light mode: one series, recessive axes.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

DESCRIPTORS = ["input_mean", "input_var", "input_absmax", "input_abs_p99", "max_over_p99",
               "kurtosis", "cancellation_ratio", "top2_gap"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def regime_histograms(cfg: dict, split: str, samples: int | None, path: Path, run_id: str) -> Path:
    pooled: dict[str, list[torch.Tensor]] = {}
    for st, i, seed in planned_samples(cfg, split):
        if samples is None or i < samples:
            pooled.setdefault(st.key, []).append(generate_stratum(st, seed))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    for ax, (key, xs) in zip(axes.flat, pooled.items()):
        x = torch.cat(xs).double()
        _style(ax)
        ax.hist(x.numpy(), bins=120, color=SERIES, edgecolor=SURFACE, linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{key}  {cfg['regimes'][key]['name']}", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("value", color=TEXT_2, fontsize=8)
        ax.set_ylabel("count (log)", color=TEXT_2, fontsize=8)
        stats = (f"mean {x.mean():.3g}\nstd {x.std():.3g}\n"
                 f"|x|max {x.abs().max():.3g}\ntensors {len(xs)}")
        ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=TEXT_2, family="monospace")
    fig.suptitle(f"Pilot input regimes: pooled FP32 values, each panel on its own x-axis  ·  run {run_id}",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def primary_error(df: pd.DataFrame) -> pd.Series:
    return pd.Series([row[PRIMARY[row["operator"]]] for _, row in df.iterrows()], index=df.index)


def pilot_medians(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(primary_metric=df["operator"].map(PRIMARY), primary_error=primary_error(df))
    g = df.groupby(["operator", "regime", "quantizer", "calib_policy"], sort=True)
    return g.agg(n_tensors=("sample_id", "nunique"), primary_metric=("primary_metric", "first"),
                 median_primary=("primary_error", "median"), median_sqnr_db=("sqnr_db", "median"),
                 median_kappa=("kappa", "median"), kappa_at_floor_rate=("kappa_at_floor", "mean"),
                 median_clip_fraction=("clip_fraction", "median"),
                 median_zero_fraction=("zero_fraction", "median"),
                 collapsed_rate=("collapsed", "mean"), top1_changed_rate=("top1_changed", "mean")).reset_index()


def predictions_check(m: pd.DataFrame, run_id: str) -> str:
    def cell(op, regime, q, policy, col):
        v = m.loc[(m.operator == op) & (m.regime == regime) & (m.quantizer == q) & (m.calib_policy == policy), col]
        return float(v.iloc[0]) if len(v) else float("nan")

    lines = [f"# Pilot check against reports/predictions.md (run {run_id})", "",
             "Pilot numbers are for debugging only and are never reported. Medians over pilot tensors.", ""]

    lines += ["## P1: D1 static, large error from clipping", "",
              "| operator | Q | static clip fraction | static primary | dynamic primary |", "| --- | --- | --- | --- | --- |"]
    for op in PRIMARY:
        for q in ("Q0", "Q1", "Q2"):
            lines.append(f"| {op} | {q} | {cell(op, 'D1', q, 'static', 'median_clip_fraction'):.4f} | "
                         f"{cell(op, 'D1', q, 'static', 'median_primary'):.3g} | "
                         f"{cell(op, 'D1', q, 'dynamic', 'median_primary'):.3g} |")

    lines += ["", "## P2: D1 dynamic Q1, most normal values become 0", "",
              f"Median zero fraction: {cell('sum', 'D1', 'Q1', 'dynamic', 'median_zero_fraction'):.3f} "
              "(predicted about 0.8 or more).", ""]

    lines += ["## P3 / P4: D1, Q2 vs Q1 (primary error ratio Q2 / Q1)", "",
              "| operator | dynamic (P3: well below 1) | static (P4: near 1) |", "| --- | --- | --- |"]
    for op in PRIMARY:
        d = cell(op, "D1", "Q2", "dynamic", "median_primary") / cell(op, "D1", "Q1", "dynamic", "median_primary")
        s = cell(op, "D1", "Q2", "static", "median_primary") / cell(op, "D1", "Q1", "static", "median_primary")
        lines.append(f"| {op} | {d:.3g} | {s:.3g} |")

    lines += ["", "## P5 / P5b / P6: D2 layer_norm", "",
              "| Q | policy | collapsed rate | median SQNR (dB) | prediction |", "| --- | --- | --- | --- | --- |"]
    expect = {("Q0", "static"): "P5 collapse", ("Q1", "static"): "P5 collapse", ("Q2", "static"): "P5 collapse",
              ("Q1", "dynamic"): "P5 collapse", ("Q2", "dynamic"): "P5b no collapse, SQNR <= 0",
              ("Q0", "dynamic"): "P6 no collapse, SQNR well under 20"}
    for (q, p), what in expect.items():
        lines.append(f"| {q} | {p} | {cell('layer_norm', 'D2', q, p, 'collapsed_rate'):.2f} | "
                     f"{cell('layer_norm', 'D2', q, p, 'median_sqnr_db'):.2f} | {what} |")

    lines += ["", "## P7: gelu amplification near 1", "",
              "| regime | Q0 dyn | Q1 dyn | Q2 dyn | Q0 static | Q1 static | Q2 static |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for regime in sorted(m.regime.unique()):
        vals = [cell("gelu", regime, q, p, "median_kappa") for p in ("dynamic", "static") for q in ("Q0", "Q1", "Q2")]
        lines.append(f"| {regime} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
    return "\n".join(lines) + "\n"


def pilot_sanity(cfg: dict, out_dir: Path, split: str, samples: int | None) -> list[Path]:
    run_id = out_dir.name
    df = pd.read_csv(out_dir / "raw" / f"{split}.csv")
    summary = out_dir / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    one_row_per_tensor = df[(df.operator == "sum") & (df.quantizer == "Q0") & (df.calib_policy == "dynamic")]
    stats = one_row_per_tensor.groupby("regime")[DESCRIPTORS].median()
    stats.insert(0, "n_tensors", one_row_per_tensor.groupby("regime")["sample_id"].nunique())
    stats_path = summary / "regime_stats.csv"
    stats.to_csv(stats_path)

    medians = pilot_medians(df)
    medians_path = summary / "pilot_medians.csv"
    medians.to_csv(medians_path, index=False)

    check_path = summary / "pilot_predictions_check.md"
    check_path.write_text(predictions_check(medians, run_id), encoding="utf-8")

    hist_path = regime_histograms(cfg, split, samples, out_dir / "plots" / "pilot_histograms.png", run_id)
    return [hist_path, stats_path, medians_path, check_path]
