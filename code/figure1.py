"""
Figure 1 — Jensen inequality (silent figure).

A: even
B: uneven (step / sin / PK) light→dark pink
C: concave
D: convex

Identical visual gaps in C/D (curve normalized to fill the panel; the
Jensen gap is a fixed *ratio* of the curve's endpoint spread, so the
on-screen dot separation matches between panels regardless of h).
No text, legend, grid, titles, or axis labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

C_EVEN = "#0021A5"
C_FE = "#0021A5"
C_EF = "#e85a9b"
C_CHORD = "#666666"
C_ENDPOINT = "#a3a3a3"
PINKS = ["#FFB000", "#FF7A00", "#FF3D00"]
PINK_CMAP = LinearSegmentedColormap.from_list("pinks_grad", PINKS)


def gradient_dot(
    ax: plt.Axes,
    x: float,
    y: float,
    ms: float,
    cmap: LinearSegmentedColormap,
    zorder: int = 7,
    n: int = 128,
) -> None:
    """Horizontal gradient: left -> right, clipped to a circle."""

    # 1D horizontal gradient
    g = np.linspace(0.0, 1.0, n)

    # (n, n, 4) RGBA image
    rgba = cmap(g)[None, :, :]
    rgba = np.repeat(rgba, n, axis=0)

    # circular alpha mask
    yy, xx = np.ogrid[-1:1:n*1j, -1:1:n*1j]
    mask = (xx**2 + yy**2) <= 1.0
    rgba[..., 3] *= mask

    # ms를 marker size(points)처럼 사용
    image = OffsetImage(
        rgba,
        zoom=ms / n,
        interpolation="bilinear",
    )

    ab = AnnotationBbox(
        image,
        (x, y),
        xycoords="data",
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=zorder,
    )
    ax.add_artist(ab)

    # white outline
    ax.plot(
        x, y, "o",
        mfc="none",
        mec="white",
        mew=0.9,
        ms=ms,
        zorder=zorder + 1,
    )

def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def silent(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def c_even(t: np.ndarray, cbar: float) -> np.ndarray:
    return np.full_like(t, cbar)


def c_step(t: np.ndarray, tau: float, cbar: float, on_frac: float = 0.5) -> np.ndarray:
    phase = np.mod(t, tau)
    return np.where(phase < on_frac * tau, cbar / on_frac, 0.0)


def c_sin(t: np.ndarray, tau: float, cbar: float, amp: float = 1.0) -> np.ndarray:
    return np.clip(cbar * (1.0 + amp * np.sin(2.0 * np.pi * t / tau)), 0.0, None)


def c_pk(t: np.ndarray, tau: float, cbar: float, k_elim: float = 1.5) -> np.ndarray:
    phase = np.mod(t, tau)
    c0 = cbar * k_elim * tau / (1.0 - np.exp(-k_elim * tau))
    return c0 * np.exp(-k_elim * phase)


def half_span_for_gap(h: float, cbar: float, target_ratio: float) -> float:
    """
    Binary search for a half-span s such that the Jensen gap, expressed as a
    FRACTION of the curve's larger endpoint deviation from f(cbar), equals
    target_ratio.

    Using a ratio (instead of a raw absolute gap, as before) matters because
    the achievable gap magnitude is on a totally different scale for e.g.
    h=0.5 vs h=2.0. Targeting a fixed absolute gap either clips the curve
    (gap tiny relative to endpoint spread -> normalized curve blows past the
    axes and gets cut down to a near-straight sliver) or never converges
    (gap unreachable, search silently saturates at the search-bound cap).
    Targeting a ratio keeps the gap a fixed proportion of the visible curve
    for any h, so C and D get the same on-screen dot separation once each
    curve is normalized to fill its own panel.
    """
    lo, hi = 1e-3 * cbar, 0.99 * cbar
    for _ in range(80):
        s = 0.5 * (lo + hi)
        c_lo, c_hi = cbar - s, cbar + s
        f_lo, f_hi, f_bar = c_lo**h, c_hi**h, cbar**h
        gap = abs(0.5 * (f_lo + f_hi) - f_bar)
        half_range = max(abs(f_hi - f_bar), abs(f_bar - f_lo))
        ratio = gap / half_range if half_range > 1e-12 else 0.0
        if ratio < target_ratio:
            lo = s
        else:
            hi = s
    return 0.5 * (lo + hi)


def jensen_panel(ax: plt.Axes, h: float, cbar: float, half_span: float) -> None:
    """
    f(c)=c^h, x normalized so [c_lo, c_hi] -> [0, 1] (\bar{c} -> 0.5), and y
    normalized so the curve's OWN endpoint spread fills the panel
    (f(\bar{c}) -> 0, the farther endpoint -> +-1). This keeps the full
    curve shape visible (fixes the earlier bug where dividing y by the much
    smaller Jensen gap pushed the endpoints far outside a fixed ylim,
    silently clipping everything but a near-straight central sliver).
    """
    c_lo = max(cbar - half_span, 1e-6 * cbar)
    c_hi = 2.0 * cbar - c_lo
    f_lo, f_hi = c_lo**h, c_hi**h
    f_bar = cbar**h
    e_f_u = 0.5 * (f_lo + f_hi)

    half_range = max(abs(f_hi - f_bar), abs(f_bar - f_lo))
    if half_range < 1e-12:
        half_range = 1.0

    def xn(c: float | np.ndarray) -> float | np.ndarray:
        return (c - c_lo) / (c_hi - c_lo)

    def yn(f: float | np.ndarray) -> float | np.ndarray:
        return (f - f_bar) / half_range

    c = np.linspace(c_lo, c_hi, 500)
    ax.plot(xn(c), yn(c**h), color="#222222", lw=2.8, zorder=3)

    ax.plot([0.0, 1.0], [yn(f_lo), yn(f_hi)], ls="--", color=C_CHORD, lw=1.4, zorder=2)
    ax.plot([0.0, 1.0], [yn(f_lo), yn(f_hi)], "o", color=C_ENDPOINT, ms=10,
            zorder=5, markeredgecolor="white", markeredgewidth=0.8)

    y_ef = float(yn(e_f_u))
    ax.vlines(0.5, min(0.0, y_ef), max(0.0, y_ef), color=C_CHORD, lw=1.6, zorder=3)
    # ax.axvline(0.5, color="#dddddd", ls=":", lw=1.0, zorder=1)
    ax.plot(0.5, 0.0, "o", color=C_FE, ms=12, zorder=5,
            markeredgecolor="white", markeredgewidth=0.9)
    gradient_dot(ax, 0.5, y_ef, ms=12, cmap=PINK_CMAP)

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("auto")
    silent(ax)


def make_figure1(
    output_path: str | Path | None = None,
    cbar: float = 1.0,
    tau: float = 3.0,
    n_cycles: int = 3,
) -> Path:
    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "figure" / "figure1" / "figure1.pdf"
    style()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    T = tau * n_cycles
    t = np.linspace(0.0, T, 1400)
    ce = c_even(t, cbar)
    cs = c_step(t, tau, cbar)
    cn = c_sin(t, tau, cbar)
    cp = c_pk(t, tau, cbar)
    ymax_c = 1.08 * max(cs.max(), cn.max(), cp.max())

    target_ratio = 0.26
    span_cv = half_span_for_gap(0.5, cbar, target_ratio)
    span_cx = half_span_for_gap(2.0, cbar, target_ratio)

    fig = plt.figure(figsize=(11.5, 4.5))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.1, 1.05, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.36, hspace=0.28,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t, ce, color=C_EVEN, lw=2.6)
    ax_a.fill_between(t, 0, ce, color=C_EVEN, alpha=0.12)
    ax_a.set_ylim(-0.05, ymax_c)
    silent(ax_a)

    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b.axhline(cbar, color="#bbbbbb", ls=":", lw=1.0)
    ax_b.plot(t, cs, color=PINKS[0], lw=2.2)
    ax_b.plot(t, cn, color=PINKS[1], lw=2.2)
    ax_b.plot(t, cp, color=PINKS[2], lw=2.2)
    ax_b.set_ylim(-0.05, ymax_c)
    silent(ax_b)

    ax_c = fig.add_subplot(gs[:, 1])
    jensen_panel(ax_c, h=0.5, cbar=cbar, half_span=span_cv)

    ax_d = fig.add_subplot(gs[:, 2])
    jensen_panel(ax_d, h=2.0, cbar=cbar, half_span=span_cx)

    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(make_figure1().resolve())

