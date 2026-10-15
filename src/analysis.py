"""Tables and figures from a run's raw CSV (plan §4-§5).

    uv run python -m src.analysis --run artifacts/<run_id>

Writes into the run folder:
  summary/summary.csv     median, IQR, p95 and bootstrap CI per operator x distribution x scheme
  summary/summary.md      the report tables: loss of precision, the sensitivity check, scheme ranking
  plots/F1_inputs.png     histograms of the input distributions
  plots/F2_error_heatmap.png  median error per distribution x scheme, one panel per operator
  plots/F3_amplification.png  kappa by operator and distribution (reductions vs the gelu control)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from src.config import DEFAULT_CONFIG, ROOT, load_config, planned_samples  # noqa: E402
from src.distributions import generate  # noqa: E402
from src.metrics import HEADLINE  # noqa: E402

# Reference palette (dataviz skill), light mode. Categorical slots in fixed order, one per operator.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OP_COLOR = {"sum": "#2a78d6", "softmax": "#eb6834", "layer_norm": "#1baf7a", "gelu": "#eda100"}
OP_MARKER = {"sum": "o", "softmax": "s", "layer_norm": "^", "gelu": "D"}
METRIC_LABEL = {"l1_norm_error": "|ΔY| / ‖A‖₁", "softmax_kl": "KL(Y ‖ Y_q), nats", "norm_l2": "‖ΔY‖₂ / ‖Y‖₂"}


def load_results(run_dir: Path) -> pd.DataFrame:
    for name in ("results.csv", "results.csv.gz"):
        if (run_dir / "raw" / name).exists():
            return pd.read_csv(run_dir / "raw" / name)
    raise FileNotFoundError(f"no raw/results.csv(.gz) in {run_dir}")


def median_ci(v: np.ndarray, rng: np.random.Generator, resamples: int, level: float) -> tuple[float, float]:
    """Percentile bootstrap CI of the median, resampling samples (tensors), not elements."""
    if len(v) == 0:
        return math.nan, math.nan
    meds = np.median(v[rng.integers(0, len(v), size=(resamples, len(v)))], axis=1)
    a = (1 - level) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def summarize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seeds"]["bootstrap_seed"])
    B, level = cfg["analysis"]["bootstrap_resamples"], cfg["analysis"]["ci_level"]
    rows = []
    for op in cfg["operators"]:
        metric = HEADLINE[op]
        for regime in cfg["regimes"]:
            for q in cfg["quantizers"]:
                g = df[(df.operator == op) & (df.regime == regime) & (df.quantizer == q)]
                v = g[metric].to_numpy(float)
                v = v[~np.isnan(v)]
                ok_kappa = g.kappa.notna() & (g.at_fp32_noise != 1)
                k = g.loc[ok_kappa, "kappa"].to_numpy(float)
                lo, hi = median_ci(v, rng, B, level)
                klo, khi = median_ci(k, rng, B, level)
                rows.append({
                    "operator": op, "regime": regime, "quantizer": q, "metric": metric,
                    "n_samples": len(g), "median": np.median(v) if len(v) else math.nan,
                    "q25": np.quantile(v, 0.25) if len(v) else math.nan,
                    "q75": np.quantile(v, 0.75) if len(v) else math.nan,
                    "p95": np.quantile(v, 0.95) if len(v) else math.nan,
                    "ci_low": lo, "ci_high": hi,
                    "median_sqnr_db": g.sqnr_db.median() if op != "sum" else math.nan,
                    "kappa_n": len(k), "kappa_excluded_fp32_noise": int((g.at_fp32_noise == 1).sum()),
                    "median_kappa": np.median(k) if len(k) else math.nan,
                    "kappa_q25": np.quantile(k, 0.25) if len(k) else math.nan,
                    "kappa_q75": np.quantile(k, 0.75) if len(k) else math.nan,
                    "kappa_ci_low": klo, "kappa_ci_high": khi,
                    "median_zero_fraction": g.zero_fraction.median(),
                    "top1_changed_rate": g.top1_changed.mean() if op == "softmax" else math.nan,
                })
    return pd.DataFrame(rows)


def ranking(s: pd.DataFrame, op: str, regime: str) -> str:
    """Schemes ordered by median error, best first, e.g. "Q0 < Q2 < Q1".

    Marked with † when even the best scheme has normalised L2 >= 0.99 (output essentially lost).
    """
    g = s[(s.operator == op) & (s.regime == regime)].sort_values("median")
    parts = [g.quantizer.iloc[0]]
    for prev, (q, m) in zip(g["median"].iloc[:-1], zip(g.quantizer.iloc[1:], g["median"].iloc[1:])):
        parts.append(("= " if m == prev else "< ") + q)
    failed = g.metric.iloc[0] == "norm_l2" and g["median"].iloc[0] >= 0.99
    return " ".join(parts) + (" †" if failed else "")


def fmt(v: float) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2e}"


def write_markdown(s: pd.DataFrame, df: pd.DataFrame, cfg: dict, manifest: dict, path: Path) -> Path:
    regimes, qs, ops = list(cfg["regimes"]), list(cfg["quantizers"]), list(cfg["operators"])
    c = manifest.get("counts", {})
    L = [f"# Quantization sensitivity results: {manifest.get('run_id', path.parent.parent.name)}", "",
         f"Commit `{manifest.get('git_commit', '?')[:7]}`{' (uncommitted changes)' if manifest.get('git_dirty') else ''}, "
         f"{manifest.get('machine', {}).get('cpu_model', '?')}. "
         f"{df.sample_id.nunique()} samples per distribution, {len(df):,} rows, "
         f"{c.get('non_finite_rows', 0)} non-finite. Medians over samples; brackets are 95% bootstrap CIs.", "",
         "Distributions: " + ", ".join(f"{r} {cfg['regimes'][r]['name']}" for r in regimes) + ".", "",
         "Schemes: " + ", ".join(f"{q} {cfg['quantizers'][q]['label']}" for q in qs) + ".", ""]

    L += ["## 1. Loss of precision against the FP32 baseline", ""]
    for op in ops:
        L += [f"**{op}**: {METRIC_LABEL[HEADLINE[op]]}", "",
              "| scheme | " + " | ".join(regimes) + " |", "| --- |" + " --- |" * len(regimes)]
        for q in qs:
            cells = []
            for r in regimes:
                x = s[(s.operator == op) & (s.regime == r) & (s.quantizer == q)].iloc[0]
                cells.append(f"{fmt(x['median'])} [{fmt(x.ci_low)}, {fmt(x.ci_high)}]")
            L.append(f"| {q} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## 2. Are reductions more sensitive than elementwise ops?", "",
          "Median amplification κ = (output relative error) / (input relative error). κ ≈ 1 means the "
          "operator passes the input's quantization error through unchanged; κ ≫ 1 means it magnifies it. "
          "The last column is the spread of the median across distributions (max / min).", ""]
    for q in qs:
        L += [f"**{q} {cfg['quantizers'][q]['label']}**", "",
              "| operator | " + " | ".join(regimes) + " | spread |", "| --- |" + " --- |" * (len(regimes) + 1)]
        for op in ops:
            meds = [s[(s.operator == op) & (s.regime == r) & (s.quantizer == q)].median_kappa.iloc[0] for r in regimes]
            valid = [m for m in meds if not math.isnan(m) and m > 0]
            spread = f"{max(valid) / min(valid):.3g}×" if len(valid) > 1 else "—"
            L.append(f"| {op} | " + " | ".join(f"{m:.3g}" if not math.isnan(m) else "—" for m in meds) + f" | {spread} |")
        L.append("")
    excl = s[s.kappa_excluded_fp32_noise > 0]
    if len(excl):
        L += ["κ values at FP32 noise level (output error within "
              f"{cfg['metrics']['fp32_noise_factor']}× of the baseline's own FP32 rounding) are excluded: "
              + ", ".join(f"{r.operator} {r.regime} {r.quantizer} ({r.kappa_excluded_fp32_noise})" for r in excl.itertuples())
              + ".", ""]

    L += ["## 3. Does the best scheme depend on the input?", "",
          "Schemes ordered by median error, best first.", "",
          "| operator | " + " | ".join(regimes) + " | distinct orders |", "| --- |" + " --- |" * (len(regimes) + 1)]
    for op in ops:
        orders = [ranking(s, op, r) for r in regimes]
        L.append(f"| {op} | " + " | ".join(orders) + f" | {len(set(o.rstrip(' †') for o in orders))} |")
    L += ["", "† every scheme has median normalised L2 ≥ 0.99: the output is essentially lost whichever scheme is used.", ""]

    soft = s[s.operator == "softmax"]
    L += ["## 4. Softmax top-1 changes", "", "Share of samples whose top-1 entry changed.", "",
          "| scheme | " + " | ".join(regimes) + " |", "| --- |" + " --- |" * len(regimes)]
    for q in qs:
        L.append(f"| {q} | " + " | ".join(f"{soft[(soft.regime == r) & (soft.quantizer == q)].top1_changed_rate.iloc[0]:.2f}"
                                          for r in regimes) + " |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_inputs(cfg: dict, samples: int, path: Path, run_id: str) -> Path:
    pooled: dict[str, list[torch.Tensor]] = {}
    for regime, params, _, seed in planned_samples(cfg, samples):
        pooled.setdefault(regime, []).append(generate(regime, cfg["vector_length"], params, seed))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    for ax, (regime, xs) in zip(axes.flat, pooled.items()):
        x = torch.cat(xs).double()
        _style(ax)
        ax.hist(x.numpy(), bins=120, color=OP_COLOR["sum"], edgecolor=SURFACE, linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{regime}  {cfg['regimes'][regime]['name']}", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("value", color=TEXT_2, fontsize=8)
        ax.set_ylabel("count (log)", color=TEXT_2, fontsize=8)
        ax.text(0.98, 0.95, f"mean {x.mean():.3g}\nstd {x.std():.3g}\n|x|max {x.abs().max():.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=TEXT_2, family="monospace")
    fig.suptitle(f"F1  Input distributions: {samples} samples × {cfg['vector_length']} values each, "
                 f"each panel on its own x-axis  ·  {run_id}", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_heatmap(s: pd.DataFrame, cfg: dict, path: Path, run_id: str, samples: int) -> Path:
    regimes, qs, ops = list(cfg["regimes"]), list(cfg["quantizers"]), list(cfg["operators"])
    fig, axes = plt.subplots(1, len(ops), figsize=(4.2 * len(ops), 4.6), facecolor=SURFACE)
    for ax, op in zip(axes, ops):
        grid = np.array([[s[(s.operator == op) & (s.regime == r) & (s.quantizer == q)]["median"].iloc[0]
                          for q in qs] for r in regimes])
        pos = grid[grid > 0]
        norm = LogNorm(vmin=pos.min(), vmax=pos.max()) if len(pos) and pos.min() < pos.max() else None
        im = ax.imshow(np.where(grid > 0, grid, np.nan), cmap="Blues", norm=norm, aspect="auto")
        for i in range(len(regimes)):
            for j in range(len(qs)):
                v = grid[i, j]
                dark = norm is not None and v > 0 and norm(v) > 0.6
                ax.text(j, i, "0" if v == 0 else f"{v:.1e}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if dark else TEXT)
        short = [f"{q}\nA{cfg['quantizers'][q]['bits']}" + (f" g{cfg['quantizers'][q]['group_size']}"
                 if cfg["quantizers"][q]["group_size"] else "") for q in qs]
        ax.set_xticks(range(len(qs)), short, fontsize=7.5, color=TEXT_2)
        ax.set_yticks(range(len(regimes)), [f"{r} {cfg['regimes'][r]['name']}" for r in regimes], fontsize=7.5, color=TEXT_2)
        ax.set_title(f"{op}\nmedian {METRIC_LABEL[HEADLINE[op]]}", color=TEXT, fontsize=9.5, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7, colors=TEXT_2)
        cb.outline.set_visible(False)
    fig.suptitle(f"F2  Loss of precision vs the FP32 baseline (median over {samples} samples; "
                 f"each operator has its own metric and colour scale)  ·  {run_id}",
                 color=TEXT, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_amplification(s: pd.DataFrame, cfg: dict, path: Path, run_id: str, samples: int) -> Path:
    regimes, qs, ops = list(cfg["regimes"]), list(cfg["quantizers"]), list(cfg["operators"])
    fig, axes = plt.subplots(1, len(qs), figsize=(5.2 * len(qs), 4.4), facecolor=SURFACE, sharey=True)
    width = 0.18
    for ax, q in zip(axes, qs):
        _style(ax)
        for k, op in enumerate(ops):
            g = s[(s.operator == op) & (s.quantizer == q)].set_index("regime").loc[regimes]
            x = np.arange(len(regimes)) + (k - (len(ops) - 1) / 2) * width
            ok = g.median_kappa.notna().to_numpy()
            yerr = np.vstack([(g.median_kappa - g.kappa_q25).to_numpy(), (g.kappa_q75 - g.median_kappa).to_numpy()])
            ax.errorbar(x[ok], g.median_kappa.to_numpy()[ok], yerr=yerr[:, ok], fmt=OP_MARKER[op], ms=6,
                        color=OP_COLOR[op], ecolor=OP_COLOR[op], elinewidth=1.5, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=op)
        ax.axhline(1, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_yscale("log")
        ax.set_xticks(range(len(regimes)), regimes, fontsize=8, color=TEXT_2)
        ax.set_title(f"{q} {cfg['quantizers'][q]['label']}", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("input distribution", color=TEXT_2, fontsize=8)
    axes[0].set_ylabel("κ = output rel. error / input rel. error (log)", color=TEXT_2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.suptitle(f"F3  Error amplification by operator: median and IQR over {samples} samples, "
                 f"FP32-noise rows excluded; dashed line κ = 1  ·  {run_id}", color=TEXT, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def analyze(run_dir: Path, cfg: dict) -> list[Path]:
    run_dir = Path(run_dir)
    df = load_results(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_id = manifest.get("run_id", run_dir.name)
    samples = int(df.sample_id.nunique())
    (run_dir / "summary").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    s = summarize(df, cfg)
    s.to_csv(run_dir / "summary" / "summary.csv", index=False)
    return [
        run_dir / "summary" / "summary.csv",
        write_markdown(s, df, cfg, manifest, run_dir / "summary" / "summary.md"),
        plot_inputs(cfg, samples, run_dir / "plots" / "F1_inputs.png", run_id),
        plot_heatmap(s, cfg, run_dir / "plots" / "F2_error_heatmap.png", run_id, samples),
        plot_amplification(s, cfg, run_dir / "plots" / "F3_amplification.png", run_id, samples),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder, or a run id under artifacts/")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.exists():
        run_dir = ROOT / "artifacts" / args.run
    for p in analyze(run_dir, load_config(args.config)):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
