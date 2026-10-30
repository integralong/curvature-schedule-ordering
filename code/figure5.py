"""
Figure 5 — distribution-free selection shift (first-order PK).

1×3 ridge plots (Lognormal | Gamma | Weibull). Each column stacks
p(x, T) over sensitivity h. Even (blue) and uneven PK (orange) sit
on the same ridge. Mean sensitivity E[X] is an "o" marker.

Baselines share mean and variance with the paper log-normal p0.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure3 import cumulative_effect, trapz

C_EVEN = "#0021A5"
C_UNEVEN = "#FF3D00"
C_P0 = "#8a8a8a"

MEAN_X = 1.0
SIGMA = 0.42
N_CYCLES = 6
TAU = 3.0
CBAR = 1.0
X_MIN = 8e-3
X_MAX = 5.0
NX = 1000

H_RIDGES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)

FAMILIES = ("Lognormal", "Gamma", "Weibull")


def configure() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _normalize(pdf: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = trapz(pdf, x)
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError("p0 normalization failed")
    return pdf / z


def lognormal_p0(x: np.ndarray, mean_x: float = MEAN_X, sigma: float = SIGMA) -> np.ndarray:
    mu = np.log(mean_x) - 0.5 * sigma**2
    pdf = (1.0 / (x * sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2.0 * sigma**2)
    )
    return _normalize(pdf, x)


def _target_moments(mean_x: float = MEAN_X, sigma: float = SIGMA) -> tuple[float, float]:
    var = (math.exp(sigma**2) - 1.0) * mean_x**2
    return mean_x, var


def gamma_p0(x: np.ndarray, mean_x: float = MEAN_X, sigma: float = SIGMA) -> np.ndarray:
    mean, var = _target_moments(mean_x, sigma)
    theta = var / mean
    k = mean / theta
    pdf = np.exp((k - 1.0) * np.log(x) - x / theta - math.lgamma(k) - k * math.log(theta))
    return _normalize(np.maximum(pdf, 0.0), x)


def _weibull_k_from_cv(cv: float) -> float:
    def f(k: float) -> float:
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        return math.sqrt(g2 / (g1 * g1) - 1.0) - cv

    lo, hi = 0.6, 20.0
    flo = f(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if flo * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = f(lo)
    return 0.5 * (lo + hi)


def weibull_p0(x: np.ndarray, mean_x: float = MEAN_X, sigma: float = SIGMA) -> np.ndarray:
    mean, var = _target_moments(mean_x, sigma)
    k = _weibull_k_from_cv(math.sqrt(var) / mean)
    lam = mean / math.gamma(1.0 + 1.0 / k)
    z = x / lam
    pdf = (k / lam) * np.power(z, k - 1.0) * np.exp(-np.power(z, k))
    return _normalize(np.maximum(pdf, 0.0), x)


P0_BUILDERS = {
    "Lognormal": lognormal_p0,
    "Gamma": gamma_p0,
    "Weibull": weibull_p0,
}


def survivor_state(x: np.ndarray, p0: np.ndarray, A: float) -> tuple[np.ndarray, float]:
    w = np.exp(-x * A)
    z = trapz(p0 * w, x)
    p = (p0 * w) / z
    ex = trapz(x * p, x)
    return p, float(ex)


def time_grid() -> np.ndarray:
    return np.logspace(math.log10(X_MIN), math.log10(X_MAX), NX)


def simulate() -> dict:
    x = time_grid()
    # Even and PK burdens are closed-form. A trapezoidal integral of the PK
    # profile crosses the redose jump and is biased (about 0.1 at h=2 on the
    # old 1800-point grid, and about 0.6 if that jump is forced onto a node).
    t_end = np.array([TAU * N_CYCLES])

    p0s = {name: fn(x) for name, fn in P0_BUILDERS.items()}
    baselines = {}
    for name, p0 in p0s.items():
        baselines[name] = {
            "p0": p0,
            "ex0": float(trapz(x * p0, x)),
        }

    ridges = {name: [] for name in FAMILIES}
    for h in H_RIDGES:
        a_e = float(cumulative_effect(t_end, h, "Even", cbar=CBAR, tau=TAU)[-1])
        a_u = float(cumulative_effect(t_end, h, "PK", cbar=CBAR, tau=TAU)[-1])
        for name, p0 in p0s.items():
            p_e, ex_e = survivor_state(x, p0, a_e)
            p_u, ex_u = survivor_state(x, p0, a_u)
            ridges[name].append({
                "h": h,
                "p_even": p_e,
                "p_uneven": p_u,
                "ex_even": ex_e,
                "ex_uneven": ex_u,
            })

    return {"x": x, "baselines": baselines, "ridges": ridges}


def _mark_mean(ax: plt.Axes, x_val: float, y_val: float, color: str, zorder: int) -> None:
    ax.plot(
        x_val, y_val, "o",
        color=color, ms=6.0, mec="black", mew=0.55, zorder=zorder,
    )


def _column_pmax(data: dict, name: str) -> float:
    pmax = float(data["baselines"][name]["p0"].max())
    for row in data["ridges"][name]:
        pmax = max(pmax, float(row["p_even"].max()), float(row["p_uneven"].max()))
    return pmax


def _style_ridge_axes(ax: plt.Axes, offsets: np.ndarray) -> None:
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_yticks(list(offsets))
    ax.set_yticklabels([r"$p_0$"] + [f"{h:g}" for h in H_RIDGES])
    ax.tick_params(
        axis="both", which="major",
        bottom=True, top=False, left=True, right=False,
        labelbottom=True, labelleft=False,
        length=3.5, labelsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def make_selection_ridge(data: dict, out_dir: Path) -> Path:
    x = data["x"]
    n_h = len(H_RIDGES)
    n_ridges = n_h + 1  # p0 at the bottom

    dy = 1.0
    offsets = np.arange(n_ridges, dtype=float) * dy

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 7.4), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, name in zip(axes, FAMILIES):
        ax.set_facecolor("white")
        p0 = data["baselines"][name]["p0"]
        ex0 = data["baselines"][name]["ex0"]
        scale = 1.85 / _column_pmax(data, name)

        layers = [{"kind": "p0", "p_even": p0, "p_uneven": None, "ex_even": ex0, "ex_uneven": None}]
        layers.extend(data["ridges"][name])

        # back to front: top ridge first so lower ridges sit in front
        for i in range(n_ridges - 1, -1, -1):
            y0 = offsets[i]
            z = 2 + (n_ridges - i)
            layer = layers[i]

            if layer.get("kind") == "p0" or layer.get("p_uneven") is None:
                p = layer["p_even"]
                y = y0 + scale * p
                ax.fill_between(x, y0, y, color="white", zorder=z, lw=0)
                ax.fill_between(x, y0, y, color=C_P0, alpha=0.20, zorder=z, lw=0)
                ax.plot(x, y, color=C_P0, ls=":", lw=1.9, zorder=z + 1, solid_capstyle="round")
                _mark_mean(ax, ex0, float(np.interp(ex0, x, y)), C_P0, z + 2)
            else:
                p_e = layer["p_even"]
                p_u = layer["p_uneven"]
                y_e = y0 + scale * p_e
                y_u = y0 + scale * p_u
                y_env = np.maximum(y_e, y_u)
                ax.fill_between(x, y0, y_env, color="white", zorder=z, lw=0)
                ax.fill_between(x, y0, y_e, color=C_EVEN, alpha=0.30, zorder=z, lw=0)
                ax.fill_between(x, y0, y_u, color=C_UNEVEN, alpha=0.30, zorder=z, lw=0)
                ax.plot(x, y_e, color=C_EVEN, lw=1.85, zorder=z + 1, solid_capstyle="round")
                ax.plot(x, y_u, color=C_UNEVEN, lw=1.85, zorder=z + 1, solid_capstyle="round")
                _mark_mean(ax, layer["ex_even"], float(np.interp(layer["ex_even"], x, y_e)), C_EVEN, z + 2)
                _mark_mean(ax, layer["ex_uneven"], float(np.interp(layer["ex_uneven"], x, y_u)), C_UNEVEN, z + 2)

        ax.set_xscale("log")
        ax.set_xlim(0.015, 3.2)
        ax.set_ylim(-0.08, offsets[-1] + 2.05)
        _style_ridge_axes(ax, offsets)
        ax.set_title(name, fontsize=12, pad=8)

    axes[0].tick_params(labelleft=True, labelbottom=True)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False, labelbottom=True)

    fig.suptitle(
        r"Final snapshot $p(x,T)$ versus curvature $h$",
        fontsize=13, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(wspace=0.12)
    out = out_dir / "figure5.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    configure()
    out_dir = Path(__file__).resolve().parents[1] / "figure" / "figure5"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = simulate()
    print(make_selection_ridge(data, out_dir))


if __name__ == "__main__":
    main()
