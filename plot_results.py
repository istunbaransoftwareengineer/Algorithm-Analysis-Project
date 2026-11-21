"""Render the benchmark's summary CSV as vector PDF and high-resolution PNG."""

import argparse
import csv
from math import isfinite
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PANELS = (
    ("fit_time_ms", "Training time"),
    ("fit_peak_rss_mib", "Training peak RAM"),
    ("fit_energy_j", "Training energy estimate"),
    ("accuracy", "Accuracy"),
    ("predict_time_ms", "Inference time"),
    ("predict_peak_rss_mib", "Inference peak RAM"),
    ("predict_energy_j", "Inference energy estimate"),
    ("f1_macro", "Macro F1"),
)
STYLES = (
    ("#2166ac", "o", "-"),
    ("#ac3a1d", "^", "--"),
    ("#167457", "s", ":"),
)
MODEL_ORDER = ("MLP", "Small CNN", "Medium CNN")
ROOT = Path(__file__).resolve().parent
WIDTH = 7.10

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.55,
    "axes.axisbelow": True,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "grid.color": "#d8dde5",
    "grid.linewidth": 0.45,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def load_summary(path):
    """Read one aggregate row per model and size."""
    grid = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            model, size = row["model"], int(row["n_train"])
            values = {}
            for field, _ in PANELS:
                mean = float(row[f"{field}_mean"])
                sd = float(row[f"{field}_sd"])
                if not isfinite(mean) or not isfinite(sd) or sd < 0:
                    raise ValueError(f"Invalid mean/SD for {field} ({model}, {size})")
                values[field] = mean, sd
            series = grid.setdefault(model, {})
            if size in series or size <= 0:
                raise ValueError(f"Invalid or duplicate training size for {model}: {size}")
            series[size] = values

    if set(grid) != set(MODEL_ORDER):
        raise ValueError(f"Expected aggregate results for {', '.join(MODEL_ORDER)}")
    sizes = sorted(grid[MODEL_ORDER[0]])
    if len(sizes) < 2 or any(set(series) != set(sizes) for series in grid.values()):
        raise ValueError("All models must have the same grid of training sizes")
    return [grid[model] for model in MODEL_ORDER], sizes


def quantity(field, observations):
    middle = median(mean for mean, _ in observations)
    unit, factor, log = "", 1, False
    if field.endswith("_time_ms"):
        unit, factor, log = ("s", 0.001, True) if middle >= 1000 else ("ms", 1, True)
    elif field.endswith("_rss_mib"):
        unit = "MiB"
    elif field.endswith("_energy_j"):
        unit, factor, log = ("mJ", 1000, True) if middle < 1 else ("J", 1, True)

    low = min(mean - sd for mean, sd in observations)
    high = max(mean + sd for mean, sd in observations)
    logarithmic = log and low > 0 and high / low >= 30
    return unit, factor, logarithmic


def errorbar(axis, x, y, xerr, yerr, style, connect):
    color, marker, line = style
    axis.errorbar(x, y, xerr=xerr, yerr=yerr, color=color,
                  marker=marker, markerfacecolor="white", markersize=3.9,
                  linewidth=0.9, elinewidth=0.65, capsize=2.4, capthick=0.65,
                  linestyle=line if connect else "none", zorder=3)


def add_legend(figure, connect):
    handles = [Line2D([], [], label=model, color=color, marker=marker,
                      markerfacecolor="white", markersize=4.5, linewidth=1,
                      linestyle=line if connect else "none")
               for model, (color, marker, line) in zip(MODEL_ORDER, STYLES)]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=3, columnspacing=2.3, handlelength=2.3)


def draw_scaling(series, sizes):
    figure, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.70))
    figure.subplots_adjust(left=0.067, right=0.974, top=0.813, bottom=0.152,
                           wspace=0.48, hspace=0.86)
    axes[1, 1].sharey(axes[0, 1])
    axes[1, 3].sharey(axes[0, 3])
    add_legend(figure, connect=True)
    ticks = [f"{size / 1000:g}k" if size >= 1000 else str(size) for size in sizes]

    for panel, (axis, (field, title)) in enumerate(zip(axes.flat, PANELS)):
        values = [[row[size][field] for size in sizes] for row in series]
        unit, factor, logarithmic = quantity(field, [pair for row in values for pair in row])
        for style, readings in zip(STYLES, values):
            means, deviations = zip(*readings)
            errorbar(axis, sizes, [value * factor for value in means], None,
                     [value * factor for value in deviations], style, connect=True)
        subtitle = unit + (" (log scale)" if logarithmic else "")
        title = f"({chr(97 + panel)}) {title}"
        axis.set_title(f"{title}\n{subtitle}" if subtitle else title, loc="left", pad=6)
        axis.set_xticks(sizes, ticks)
        axis.set_axisbelow(True)
        axis.grid(axis="y")
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.yaxis.set_major_locator(MaxNLocator(4))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    figure.supxlabel("Training examples", y=0.04, fontsize=8)
    return figure


def draw_tradeoffs(series, size):
    figure, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05), sharey=True)
    figure.subplots_adjust(left=0.068, right=0.974, top=0.708, bottom=0.235, wspace=0.27)
    add_legend(figure, connect=False)
    rows = [row[size] for row in series]

    for panel, (axis, (field, title)) in enumerate(zip(axes, PANELS[4:7])):
        costs = [row[field] for row in rows]
        unit, factor, logarithmic = quantity(field, costs)
        for style, row, (cost, cost_sd) in zip(STYLES, rows, costs):
            score, score_sd = row["f1_macro"]
            errorbar(axis, cost * factor, score, cost_sd * factor, score_sd,
                     style, connect=False)
        axis.set_title(f"({chr(97 + panel)}) Macro F1", loc="left", pad=6)
        unit_label = f"{unit}, log scale" if logarithmic else unit
        axis.set_xlabel(f"{title} ({unit_label})")
        axis.grid(axis="y")
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.tick_params(axis="y", labelleft=True)
        if logarithmic:
            axis.set_xscale("log")
        else:
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    series, sizes = load_summary(args.summary)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = (("scaling", draw_scaling(series, sizes)),
               ("tradeoffs", draw_tradeoffs(series, sizes[-1])))
    for name, figure in figures:
        figure.savefig(args.out / f"{name}.pdf")
        figure.savefig(args.out / f"{name}.png", dpi=320)
        plt.close(figure)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
