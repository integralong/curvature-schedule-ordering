"""
Shared helpers for equal-mean even vs uneven tumor-burden figures.

Schedules: step, sinusoidal, PK
Styles:    detailed (axes/labels) and silent (curves only)
Colors:    even = blue solid, uneven = orange solid (no dash)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

C_EVEN = "#0021A5"
C_UNEVEN = "#FF3D00"

Sched = Callable[[np.ndarray], np.ndarray]


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def trapz_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.trapezoid(y, x, axis=1) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=1)


def cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))
    return out


def lognormal_p0(x: np.ndarray, mean_x: float = 1.0, sigma: float = 0.42) -> np.ndarray:
    mu = np.log(mean_x) - 0.5 * sigma**2
    pdf = (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2 * sigma**2)
    )
    return pdf / trapz(pdf, x)


def schedule_even(t: np.ndarray, cbar: float = 1.0) -> np.ndarray:
    return np.full_like(t, cbar)


def schedule_step(
    t: np.ndarray,
    cbar: float = 1.0,
    tau: float = 3.0,
    on_frac: float = 0.5,
) -> np.ndarray:
    phase = np.mod(t, tau)
    return np.where(phase < on_frac * tau, cbar / on_frac, 0.0)


def schedule_sinu(
    t: np.ndarray,
    cbar: float = 1.0,
    tau: float = 3.0,
    amp: float = 1.0,
) -> np.ndarray:
    return np.clip(cbar * (1.0 + amp * np.sin(2.0 * np.pi * t / tau)), 0.0, None)


def schedule_pk(
    t: np.ndarray,
    cbar: float = 1.0,
    tau: float = 3.0,
    k_elim: float = 1.5,
) -> np.ndarray:
    phase = np.mod(t, tau)
    c0 = cbar * k_elim * tau / (1.0 - np.exp(-k_elim * tau))
    return c0 * np.exp(-k_elim * phase)


UNEIVEN_FAMILIES: list[tuple[str, Sched]] = [
    ("Step", schedule_step),
    ("Sinusoidal", schedule_sinu),
    ("PK", schedule_pk),
]


def cumulative_effect(t: np.ndarray, h: float, family: str, *,
                      cbar: float = 1.0, tau: float = 3.0,
                      on_frac: float = 0.5, amp: float = 1.0,
                      k_elim: float = 1.5) -> np.ndarray:
    """Integrated c(t)**h. Step and PK use exact piecewise integrals.

    For the sinusoidal profile, integrate on a uniform time grid, which
    includes all cycle boundaries. This smooth integral is convergent.
    """
    t = np.asarray(t, dtype=float)
    if h <= 0 or tau <= 0 or cbar <= 0:
        raise ValueError("h, tau, and cbar must be positive")
    cycles = np.floor(t / tau).astype(int)
    phase = np.mod(t, tau)
    if family == "Even":
        return t * cbar**h
    if family == "Step":
        if not (0 < on_frac <= 1):
            raise ValueError("on_frac must be in (0, 1]")
        high = (cbar / on_frac)**h
        return high * (cycles * on_frac * tau + np.minimum(phase, on_frac * tau))
    if family == "PK":
        c0 = cbar * k_elim * tau / (-np.expm1(-k_elim * tau))
        factor = c0**h / (h * k_elim)
        return factor * (cycles * (-np.expm1(-h*k_elim*tau))
                         + (-np.expm1(-h*k_elim*phase)))
    if family == "Sinusoidal":
        if not 0 <= amp <= 1:
            raise ValueError("amp must be between zero and one")
        c = schedule_sinu(t, cbar=cbar, tau=tau, amp=amp)
        return cumtrapz(c**h, t)
    raise ValueError(f"Unknown schedule family: {family}")


def simulate_pair(
    h: float,
    uneven_sched: Sched,
    *,
    n_cycles: int = 6,
    tau: float = 3.0,
    N0: float = 1e9,
    gamma: float = 0.0,
    sigma: float = 0.42,
    mean_x: float = 1.0,
    x_min: float = 0.05,
    x_max: float = 8.0,
    nx: int = 700,
    nt_cycle: int = 300,
) -> dict:
    T = tau * n_cycles
    # +1 gives exactly nt_cycle equal intervals per cycle.
    t = np.linspace(0.0, T, nt_cycle * n_cycles + 1)
    x = np.logspace(np.log10(x_min), np.log10(x_max), nx)
    p0 = lognormal_p0(x, mean_x, sigma)
    families = {schedule_step: "Step", schedule_sinu: "Sinusoidal",
                schedule_pk: "PK"}
    if uneven_sched not in families:
        raise ValueError("Expected one of the three built-in uneven schedules")
    family = families[uneven_sched]

    def N_of(A: np.ndarray) -> tuple[np.ndarray, float]:
        Z = trapz_rows(p0[None, :] * np.exp(-np.outer(A, x)), x)
        N = N0 * np.exp(gamma * t) * Z
        return N, float(N[-1])

    A_even = cumulative_effect(t, h, "Even", tau=tau)
    A_uneven = cumulative_effect(t, h, family, tau=tau)
    Ne, Ne_f = N_of(A_even)
    Nu, Nu_f = N_of(A_uneven)
    return {
        "t": t, "N_even": Ne, "N_uneven": Nu,
        "N_even_final": Ne_f, "N_uneven_final": Nu_f,
        "A_even_final": float(A_even[-1]),
        "A_uneven_final": float(A_uneven[-1]),
        "tau": tau, "n_cycles": n_cycles, "h": h,
    }


def silent_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both", which="both",
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False,
    )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)


def plot_one_panel(
    ax: plt.Axes,
    data: dict,
    schedule_name: str,
    *,
    detailed: bool,
    show_legend: bool,
) -> None:
    t = data["t"]
    ax.semilogy(t, data["N_even"], color=C_EVEN, lw=2.4, solid_capstyle="round")
    ax.semilogy(t, data["N_uneven"], color=C_UNEVEN, lw=2.4, solid_capstyle="round")

    y = np.concatenate([data["N_even"], data["N_uneven"]])
    y = y[np.isfinite(y) & (y > 0)]
    ax.set_ylim(y.min() * 0.7, y.max() * 1.15)
    ax.set_xlim(float(t[0]), float(t[-1]))

    if not detailed:
        silent_axes(ax)
        return

    tau = float(data["tau"])
    for k in range(int(data["n_cycles"]) + 1):
        ax.axvline(k * tau, color="#dddddd", lw=0.7, zorder=0)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$N(t)$")
    ax.set_title(schedule_name, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.text(
        0.03, 0.04,
        rf"$N_{{\mathrm{{e}}}}={data['N_even_final']:.2e}$"
        + "\n"
        + rf"$N_{{\mathrm{{u}}}}={data['N_uneven_final']:.2e}$",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        color="#333333",
    )

    if show_legend:
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=C_EVEN, lw=2.3, label="even"),
                plt.Line2D([0], [0], color=C_UNEVEN, lw=2.3, label="uneven"),
            ],
            fontsize=8,
            loc="upper right",
            frameon=False,
        )


def make_burden_figure(
    h: float,
    *,
    detailed: bool,
    out_dir: Path,
    stem: str,
) -> Path:
    """Three rows: Step, Sinusoidal, PK. Even=blue solid; uneven=orange solid."""
    style = "detailed" if detailed else "silent"
    out = out_dir / f"{stem}_{style}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(5.4, 9.6) if detailed else (4.8, 9.0),
        sharex=True,
    )
    fig.patch.set_facecolor("white")

    winner = "even" if h < 1.0 else "uneven"
    if detailed:
        fig.suptitle(
            rf"$N(t)$ under equal-mean dose  ($h={h:g}$; lower burden: {winner})",
            fontsize=12,
            y=0.995,
        )

    for i, (name, sched) in enumerate(UNEIVEN_FAMILIES):
        ax = axes[i]
        ax.set_facecolor("white")
        data = simulate_pair(h, sched)
        plot_one_panel(
            ax, data, name,
            detailed=detailed,
            show_legend=(detailed and i == 0),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.97) if detailed else (0, 0, 1, 1))
    fig.savefig(
        out,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.04 if not detailed else 0.12,
    )
    plt.close(fig)
    return out


def configure() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
    })


def main() -> None:
    configure()
    out = Path(__file__).resolve().parents[1] / "figure" / "figure3"
    for h, stem in [(0.5, "figure3_h0p5"), (2.0, "figure3_h2")]:
        for detailed in (True, False):
            path = make_burden_figure(h, detailed=detailed, out_dir=out, stem=stem)
            print(path)
        for label, schedule in UNEIVEN_FAMILIES:
            result = simulate_pair(h, schedule)
            print(f"h={h:g} {label:10s} A_even={result['A_even_final']:.6f} "
                  f"A_uneven={result['A_uneven_final']:.6f} "
                  f"N_even={result['N_even_final']:.6g} "
                  f"N_uneven={result['N_uneven_final']:.6g}")


if __name__ == "__main__":
    main()
