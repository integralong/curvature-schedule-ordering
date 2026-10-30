"""
Figure 4 — selection under equal-mean schedules.

Writes two PDF families into figure/figure4/:
  figure4_mean_h*.pdf         E[X] at each cycle boundary
  figure4_resistance_h*.pdf   final p(x, T) for h=0.5 and h=2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure3 import cumulative_effect, lognormal_p0, trapz

C_EVEN = "#0021A5"
C_UNEVEN = "#FF3D00"
C_BASE = "#aaaaaa"
C_P0 = "#222222"
C_STEP = "#FFB000"
C_SIN = "#FF7A00"
C_PK = "#FF3D00"

MEAN_FAMILIES = ("Step", "Sinusoidal", "PK")
RESISTANCE_FAMILIES = [
    ("Step", C_STEP),
    ("Sinusoidal", C_SIN),
    ("PK", C_PK),
]


def _out_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "figure" / "figure4"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _htag(h: float) -> str:
    return f"{h:g}".replace(".", "p")


def configure() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def mean_sensitivity_by_cycle(
    h: float,
    family: str,
    *,
    mean_x: float = 0.6,
    sigma: float = 0.42,
    n_cycles: int = 6,
    tau: float = 3.0,
    cbar: float = 1.0,
    x_min: float = 0.03,
    x_max: float = 5.0,
    nx: int = 900,
    nt_cycle: int = 300,
) -> dict:
    """E[X] among survivors at each cycle boundary 0..n_cycles."""
    T = tau * n_cycles
    t = np.linspace(0.0, T, nt_cycle * n_cycles + 1)
    x = np.logspace(np.log10(x_min), np.log10(x_max), nx)
    p0 = lognormal_p0(x, mean_x, sigma)
    idxs = np.arange(0, n_cycles + 1) * nt_cycle

    def ex_trajectory(which: str) -> np.ndarray:
        A = cumulative_effect(t, h, which, cbar=cbar, tau=tau)
        ex = []
        for i in idxs:
            Ai = float(A[i])
            w = np.exp(-x * Ai)
            Z = trapz(p0 * w, x)
            p = (p0 * w) / Z
            ex.append(float(trapz(x * p, x)))
        return np.asarray(ex, dtype=float)

    return {
        "cycles": np.arange(0, n_cycles + 1),
        "ex_even": ex_trajectory("Even"),
        "ex_uneven": ex_trajectory(family),
        "ex0": float(trapz(x * p0, x)),
        "h": h,
    }


def make_mean_sensitivity_figure(h: float, *, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or _out_dir()
    out = out_dir / f"figure4_mean_h{_htag(h)}.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, name in zip(axes, MEAN_FAMILIES):
        data = mean_sensitivity_by_cycle(h, name)
        cyc = data["cycles"]

        ax.axhline(data["ex0"], color=C_BASE, ls=":", lw=1.2, zorder=0)
        ax.plot(
            cyc, data["ex_even"],
            color=C_EVEN, lw=2.3, marker="o", ms=6, solid_capstyle="round",
        )
        ax.plot(
            cyc, data["ex_uneven"],
            color=C_UNEVEN, lw=2.3, marker="o", ms=6, solid_capstyle="round",
        )

        ax.set_xticks(cyc)
        ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        ax.set_ylim(0.15, 0.62)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def survivor_state(
    x: np.ndarray,
    p0: np.ndarray,
    A: float,
) -> tuple[np.ndarray, float]:
    w = np.exp(-x * A)
    Z = trapz(p0 * w, x)
    p = (p0 * w) / Z
    ex = trapz(x * p, x)
    return p, float(ex)


def run_final_distributions(
    h: float,
    *,
    mean_x: float = 0.6,
    sigma: float = 0.42,
    n_cycles: int = 6,
    tau: float = 3.0,
    cbar: float = 1.0,
    x_min: float = 0.03,
    x_max: float = 5.0,
    nx: int = 900,
    nt_cycle: int = 300,
) -> dict:
    T = tau * n_cycles
    t = np.linspace(0.0, T, nt_cycle * n_cycles + 1)
    x = np.logspace(np.log10(x_min), np.log10(x_max), nx)
    p0 = lognormal_p0(x, mean_x, sigma)
    ex0 = float(trapz(x * p0, x))

    def final_of(family: str) -> tuple[np.ndarray, float, float]:
        A_T = float(cumulative_effect(t, h, family, cbar=cbar, tau=tau)[-1])
        pT, exT = survivor_state(x, p0, A_T)
        return pT, exT, A_T

    p_even, ex_even, A_even = final_of("Even")

    out = {
        "T": T,
        "x": x,
        "p0": p0,
        "ex0": ex0,
        "p_even": p_even,
        "ex_even": ex_even,
        "A_even": A_even,
        "uneven": {},
    }
    for name, color in RESISTANCE_FAMILIES:
        p_u, ex_u, A_u = final_of(name)
        out["uneven"][name] = {"p": p_u, "ex": ex_u, "A": A_u, "color": color}
    return out


def _mark_mean(ax: plt.Axes, x_val: float, p: np.ndarray, x: np.ndarray, color: str) -> None:
    y = float(np.interp(x_val, x, p))
    ax.plot(x_val, y, "o", color=color, ms=7, mec="black", mew=0.8, zorder=5)


def make_resistance_figure(h: float, *, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or _out_dir()
    data = run_final_distributions(h)
    x = data["x"]
    out = out_dir / f"figure4_resistance_h{_htag(h)}.pdf"

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(x, data["p0"], color=C_P0, ls=":", lw=2.2)
    ax.plot(x, data["p_even"], color=C_EVEN, lw=2.5)
    _mark_mean(ax, data["ex_even"], data["p_even"], x, C_EVEN)

    for name, color in RESISTANCE_FAMILIES:
        u = data["uneven"][name]
        ax.plot(x, u["p"], color=color, lw=2.3)
        _mark_mean(ax, u["ex"], u["p"], x, color)

    ax.set_xscale("log")
    ax.set_xlim(float(x.min()), min(float(x.max()), 4.0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    configure()
    out_dir = _out_dir()
    for h in (0.5, 2.0):
        print(make_mean_sensitivity_figure(h, out_dir=out_dir))
        print(make_resistance_figure(h, out_dir=out_dir))


if __name__ == "__main__":
    main()
