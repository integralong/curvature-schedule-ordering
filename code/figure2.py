"""
Figure 2 — Homogeneous / discrete / continuous sensitivity.

Columns:
  1. p0(x)               — the baseline sensitivity distribution
  2. per-phenotype kill   — K = x A (always a straight line, one per phenotype)
  3. population survival  — S(A) = E_p0[e^{-X A}]  (the p0-weighted average
                             of the individual exponentials in col 2; this is
                             the quantity that actually appears in N(T), and
                             it is where "apparent curvature" from averaging
                             over heterogeneity first shows up)

Rows:
  1. homogeneous  — delta p0(x)      | single log-kill line | pure exponential decay
  2. discrete     — two spikes       | two log-kill lines    | bi-exponential (kinked) decay
  3. continuous   — density          | fan of log-kill lines | smoothly-curving decay

No text, labels, legend, title, or grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

C_HOMO = "gray"
C_DISC = ["#B40426", "#3B4CC0"]
CMAP_CONT = plt.cm.coolwarm_r  # low x (resistant) → red; high x → blue
C_SURV = "#222222"             # population-level survival: always dark, regardless of row
C_REF = "#999999"              # dashed reference: constant-rate ("straight") decay chord


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.dpi": 800,
        "savefig.bbox": None,
        "savefig.facecolor": "white",
    })


def silent(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)



SPINE_LW = 3.4


def thick_axes(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float,
               logy: bool = False) -> None:
    """Thick L-shaped axes (no arrowheads, no ticks, no text)."""
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_position(("data", xmin))
    ax.spines["bottom"].set_position(("data", ymin))
    ax.spines["left"].set_linewidth(SPINE_LW)
    ax.spines["bottom"].set_linewidth(SPINE_LW)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_box_aspect(1)
    silent(ax)
    ax.axis("off")


def spike(ax: plt.Axes, x0: float, height: float, color: str, lw: float = 2.4) -> None:
    ax.vlines(x0, 0.0, height, color=color, lw=lw)


def logkill_line(ax: plt.Axes, x_sens: float, A: np.ndarray, color: str, lw: float = 2.0) -> None:
    # K = x * A  (cumulative log-kill)
    ax.plot(A, x_sens * A, color=color, lw=lw)


def ref_chord(ax: plt.Axes, A_pos: np.ndarray, S: np.ndarray) -> None:
    """Dashed chord connecting the first/last points only. On a log-y axis,
    matplotlib connects the two vertices with a straight segment in the
    (already log-transformed) screen space, so this chord IS the constant-rate
    ("pure exponential") decay between the same endpoints. Any bow of the
    solid aggregate curve away from this dashed line is the curvature that
    averaging over heterogeneity introduces."""
    ax.plot([A_pos[0], A_pos[-1]], [S[0], S[-1]], ls="--", color=C_REF, lw=2.0, zorder=3)


def make_figure2(
    output_path: str | Path | None = None,
) -> Path:
    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "figure" / "figure2" / "figure2.pdf"
    style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # shared panel windows so every column lines up
    x_max = 2.6
    y_p0 = 1.25
    A_max = 2.4
    k_max = 2.6
    A_surv_max = 2.4

    x0 = 1.0
    x1, x2 = 0.30, 1.90
    w = 0.55
    A = np.linspace(0.0, A_max, 200)
    A_pos = np.linspace(1e-3, A_surv_max, 240)

    x = np.linspace(1e-3, x_max, 500)
    mu, sig = np.log(1.0), 0.60
    p0 = (1.0 / (x * sig * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sig**2))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p0 = p0 / trapz(p0, x)
    p0_plot = p0 * (0.92 * y_p0 / p0.max())

    x_fan = np.linspace(0.20, 2.4, 14)
    norm = Normalize(vmin=x_fan.min(), vmax=x_fan.max())

    S_homo = np.exp(-x0 * A_pos)
    S_disc = w * np.exp(-x1 * A_pos) + (1.0 - w) * np.exp(-x2 * A_pos)
    S_cont = np.array([trapz(p0 * np.exp(-x * a), x) for a in A_pos])

    s_min = 0.8 * min(S_homo.min(), S_disc.min(), S_cont.min())
    s_max = 1.15

    fig = plt.figure(figsize=(10.8, 10.8))
    gs = GridSpec(
        3, 3, figure=fig,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.06, right=0.97, top=0.97, bottom=0.06,
        hspace=0.22, wspace=0.22,
    )

    # ------------------------------------------------------------------ row 1 homogeneous
    ax = fig.add_subplot(gs[0, 0])
    thick_axes(ax, 0.0, x_max, 0.0, y_p0)
    spike(ax, x0, 1.0, C_HOMO, lw=10)

    ax = fig.add_subplot(gs[0, 1])
    thick_axes(ax, 0.0, A_max, 0.0, k_max)
    logkill_line(ax, x0, A, C_HOMO, lw=5)

    ax = fig.add_subplot(gs[0, 2])
    thick_axes(ax, 0.0, A_surv_max, s_min, s_max, logy=True)
    ax.plot(A_pos, S_homo, color=C_SURV, lw=5)

    # ------------------------------------------------------------------ row 2 discrete
    ax = fig.add_subplot(gs[1, 0])
    thick_axes(ax, 0.0, x_max, 0.0, y_p0)
    spike(ax, x1, w, C_DISC[0], lw=10)
    spike(ax, x2, 1.0 - w, C_DISC[1], lw=10)

    ax = fig.add_subplot(gs[1, 1])
    thick_axes(ax, 0.0, A_max, 0.0, k_max)
    logkill_line(ax, x1, A, C_DISC[0], lw=5)
    logkill_line(ax, x2, A, C_DISC[1], lw=5)

    ax = fig.add_subplot(gs[1, 2])
    thick_axes(ax, 0.0, A_surv_max, s_min, s_max, logy=True)
    ax.plot(A_pos, np.exp(-x1 * A_pos), color=C_DISC[0], lw=4)
    ax.plot(A_pos, np.exp(-x2 * A_pos), color=C_DISC[1], lw=4)
    ref_chord(ax, A_pos, S_disc)
    ax.plot(A_pos, S_disc, color=C_SURV, lw=3.4, zorder=6)

    # ------------------------------------------------------------------ row 3 continuous
    ax = fig.add_subplot(gs[2, 0])
    thick_axes(ax, 0.0, x_max, 0.0, y_p0)
    for i in range(len(x) - 1):
        col = CMAP_CONT(norm(0.5 * (x[i] + x[i + 1])))
        ax.fill_between(x[i:i + 2], 0.0, p0_plot[i:i + 2], color=col, linewidth=0)

    ax = fig.add_subplot(gs[2, 1])
    thick_axes(ax, 0.0, A_max, 0.0, k_max)
    for xv in x_fan:
        logkill_line(ax, float(xv), A, CMAP_CONT(norm(xv)), lw=4)

    ax = fig.add_subplot(gs[2, 2])
    thick_axes(ax, 0.0, A_surv_max, s_min, s_max, logy=True)
    for xv in x_fan:
        ax.plot(A_pos, np.exp(-xv * A_pos), color=CMAP_CONT(norm(xv)), lw=1.7)
    ref_chord(ax, A_pos, S_cont)
    ax.plot(A_pos, S_cont, color=C_SURV, lw=4, zorder=6)

    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(make_figure2().resolve())