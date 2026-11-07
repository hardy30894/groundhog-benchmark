"""Draws the README's charts from the result files.

pip install matplotlib
python scripts/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tables  # noqa: E402
from paper import TABLE1  # noqa: E402

from groundhog.conditions import CONDITIONS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
GROSS = "#c3d3e6"
EARNED = "#2f5d8c"
WRONG = "#c0392b"

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.edgecolor": GRID,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def savings():
    """Reported and earned savings per system, with the pairs where it made a wrong write."""
    systems = list(TABLE1)[::-1]
    stats = [tables.summary(tables.rows(TABLE1[s])) for s in systems]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = range(len(systems))
    ax.barh(
        [i + 0.19 for i in y],
        [s["gross"] for s in stats],
        height=0.36,
        color=GROSS,
        label="reported (gross) savings",
    )
    ax.barh(
        [i - 0.19 for i in y],
        [s["earned"] for s in stats],
        height=0.36,
        color=EARNED,
        label="earned savings: only runs that came out right",
    )
    for i, s in enumerate(stats):
        ax.text(
            max(s["gross"], 0) + 1.5,
            i + 0.19,
            f"{s['gross']:.1f}%",
            va="center",
            fontsize=9,
            color=MUTED,
        )
        ax.text(
            max(s["earned"], 0) + 1.5,
            i - 0.19,
            f"{s['earned']:.1f}%",
            va="center",
            fontsize=9,
            color=EARNED,
            fontweight="bold",
        )
        ax.text(
            1.02,
            (i + 0.5) / len(systems),
            f"{s['failed']}/33",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=10,
            color=WRONG if s["failed"] else MUTED,
        )
    ax.text(
        1.02,
        1.0,
        "pairs with a\nwrong write",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.set_yticks(list(y), systems)
    ax.set_ylim(-0.6, len(systems) - 0.4)
    ax.set_xlim(-20, 105)
    ax.set_xticks([-20, 0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{t}%" for t in (-20, 0, 20, 40, 60, 80, 100)])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title(
        "Tokens saved against running without a cache, 3,300 instances",
        loc="left",
        fontsize=12,
        pad=28,
    )
    fig.tight_layout()
    fig.savefig(OUT / "savings.svg")
    plt.close(fig)


def families():
    """Earned savings per family and system, with wrong writes marked."""
    order = [f for members in CONDITIONS.values() for f in members]
    systems = list(TABLE1)
    grid, wrong = [], []
    for family in order:
        row, marks = [], []
        for system in systems:
            rs = tables.family(tables.rows(TABLE1[system]), family)
            row.append(tables.pooled(rs))
            marks.append(sum(r["gate"] == "FAIL" for r in rs))
        grid.append(row)
        wrong.append(marks)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, family in enumerate(order):
        for j in range(len(systems)):
            value = grid[i][j]
            colour = "white" if value > 55 else INK
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=10, color=colour)
            if wrong[i][j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.46, i - 0.44), 0.92, 0.88, fill=False, edgecolor=WRONG, linewidth=1.6
                    )
                )
                ax.text(
                    j + 0.38,
                    i - 0.27,
                    str(wrong[i][j]),
                    ha="right",
                    va="center",
                    fontsize=7,
                    color=WRONG,
                )
    ax.set_xticks(range(len(systems)), systems, fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    boundary = 0
    for condition, members in CONDITIONS.items():
        middle = boundary + (len(members) - 1) / 2
        ax.text(
            -1.75,
            middle,
            condition,
            ha="right",
            va="center",
            fontsize=10,
            color=MUTED,
            style="italic",
        )
        boundary += len(members)
        if boundary < len(order):
            ax.axhline(boundary - 0.5, color="white", linewidth=4)
    ax.set_xlim(-0.5, len(systems) - 0.5)
    fig.text(
        0.02,
        0.02,
        "Earned savings in percent, three domains pooled. A red border marks a "
        "family with a wrong write;\nits number is how many of the three domains had one.",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.04, 0.07, 1, 1))
    fig.savefig(OUT / "families.svg")
    plt.close(fig)


def latency():
    """AgenticCache's savings as the model's check arrives later."""
    ks = (1, 2, 3, 5, 10)
    stats = [tables.summary(tables.rows(f"agenticcache/k{k}")) for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    ax.plot(
        ks,
        [s["gross"] for s in stats],
        marker="o",
        color=GROSS,
        linewidth=2.5,
        label="reported (gross)",
    )
    ax.plot(
        ks, [s["earned"] for s in stats], marker="o", color=EARNED, linewidth=2.5, label="earned"
    )
    for k, s in zip(ks, stats):
        gross_above = s["gross"] >= s["earned"]
        ax.text(
            k + 0.15,
            s["gross"] + (4 if gross_above else -9),
            f"{s['gross']:.1f}%",
            ha="left",
            fontsize=9,
            color=MUTED,
        )
        ax.text(
            k + 0.15,
            s["earned"] + (-9 if gross_above else 4),
            f"{s['earned']:.1f}%",
            ha="left",
            fontsize=9,
            color=EARNED,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(ks, [f"K = {k}" for k in ks])
    ax.set_xlim(0.5, 11.2)
    ax.set_ylim(-25, 75)
    ax.set_yticks([-20, 0, 20, 40, 60])
    ax.set_yticklabels([f"{t}%" for t in (-20, 0, 20, 40, 60)])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(
        "AgenticCache: the later the model's check arrives, the more is reported saved",
        loc="left",
        fontsize=11,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "latency.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    savings()
    families()
    latency()
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
