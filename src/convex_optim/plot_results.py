"""从保存的逐次迭代数据生成课程作业图，不重新运行优化算法。

本模块只用 NumPy 与 Matplotlib。等高线使用题目中的函数，曲线与轨迹
使用 JSON 中的实测记录；第一题的解析曲线单独作为核验参照。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#245C9F"
ORANGE = "#D47C24"
GREEN = "#31856C"
PURPLE = "#8463A4"
GRAY = "#5B6877"
LIGHT = "#BBC4CE"
INK = "#203344"
F_STAR = float(2.0 * np.sqrt(2.0) * np.exp(-0.1))
X_STAR = np.array([-0.5 * np.log(2.0), 0.0])

STYLES = [
    {"color": BLUE, "linestyle": "-", "marker": "o"},
    {"color": ORANGE, "linestyle": "--", "marker": "s"},
    {"color": GREEN, "linestyle": "-.", "marker": "^"},
    {"color": PURPLE, "linestyle": ":", "marker": "D"},
]


def _style() -> dict[str, Any]:
    """少量颜色与不同线型兼顾屏幕阅读和黑白打印。"""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.2,
        "axes.titleweight": "medium",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#748291",
        "axes.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": "#455869",
        "ytick.color": "#455869",
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D5DDE5",
        "grid.color": "#DFE5EB",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _values(run: dict, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in run["history"]], dtype=float)


def _updates(run: dict) -> int:
    return int(run["summary"].get("iterations", run["history"][-1]["k"]))


def _lookup(runs: list[dict], problem: str, **criteria: Any) -> dict:
    for run in runs:
        if run["problem"] != problem:
            continue
        matches = True
        for key, expected in criteria.items():
            actual = run.get(key)
            if isinstance(expected, (int, float)):
                matches &= actual is not None and np.isclose(
                    float(actual), float(expected), rtol=1e-10, atol=1e-13
                )
            else:
                matches &= actual == expected
        if matches:
            return run
    raise ValueError(f"Missing run: problem={problem!r}, criteria={criteria!r}")


def _find_exact(runs: list[dict]) -> dict:
    for run in runs:
        if run["problem"] == "exponential" and run["method"] in {
            "exact", "exact_bisection"
        }:
            return run
    raise ValueError("Exponential exact-search run is missing.")


def _nonquadratic_style(run: dict) -> dict:
    """让重复出现的基准组在不同图中保持相同颜色与线型。"""
    if run["method"] in {"exact", "exact_bisection"}:
        return STYLES[0]
    alpha, beta = float(run["alpha"]), float(run["beta"])
    if np.isclose(beta, 0.7):
        return STYLES[3] if np.isclose(alpha, 0.01) else STYLES[1] if np.isclose(alpha, 0.1) else STYLES[2]
    if np.isclose(beta, 0.3):
        return STYLES[2]
    if np.isclose(beta, 0.5):
        return STYLES[0]
    return STYLES[3]


def _save(fig: plt.Figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", pad_inches=0.10, **kwargs)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def _clean_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.6)
    if grid:
        ax.grid(True, which="major", zorder=0)


def _line(
    ax: plt.Axes,
    run: dict,
    key: str,
    style: dict,
    label: str | None = None,
    *,
    log: bool = False,
    markers: bool = True,
) -> None:
    k = _values(run, "k")
    y = _values(run, key)
    good = np.isfinite(y)
    if log:
        good &= y > 0.0
        ax.set_yscale("log")
    options = {
        "color": style["color"],
        "linestyle": style["linestyle"],
        "label": label,
        "zorder": 3,
    }
    if markers:
        options.update(
            marker=style["marker"],
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=0.85,
            markevery=max(1, int(np.count_nonzero(good)) // 16),
        )
    ax.plot(k[good], y[good], **options)


def _trajectory(
    ax: plt.Axes, run: dict, color: str, optimum: np.ndarray, *, annotate_first: bool = False
) -> None:
    pts = np.column_stack((_values(run, "x1"), _values(run, "x2")))
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.15, zorder=4)
    early = pts[: min(14, len(pts))]
    ax.scatter(
        early[:, 0], early[:, 1], s=16, facecolors="white", edgecolors=color,
        linewidths=0.85, zorder=5,
    )
    ax.scatter(*pts[0], s=44, facecolors="white", edgecolors=INK, linewidths=1.05, zorder=7)
    ax.scatter(*optimum, marker="*", s=95, facecolor=INK, edgecolor="white", linewidth=0.6, zorder=8)
    ax.annotate(
        r"$x^{(0)}$", pts[0], xytext=(6, 9), textcoords="offset points",
        fontsize=10, color=INK,
    )
    ax.annotate(
        r"$x^\star$", optimum, xytext=(-16, -16), textcoords="offset points",
        fontsize=11, color=INK,
    )
    if annotate_first and len(pts) > 1:
        ax.annotate(
            r"$x^{(1)}$", pts[1], xytext=(6, -17), textcoords="offset points",
            fontsize=9.0, color=color,
        )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")
    _clean_axes(ax, grid=False)


def _quadratic_paths(runs: list[dict], output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), layout="constrained")
    cases = [(1.0, 1.65), (10.0, 11.8), (0.1, 1.18)]
    for i, (ax, (gamma, radius)) in enumerate(zip(axes, cases)):
        run = _lookup(runs, "quadratic", gamma=gamma)
        z = np.linspace(-radius, radius, 480)
        xx, yy = np.meshgrid(z, z)
        zz = 0.5 * (xx**2 + gamma * yy**2)
        f0 = float(run["history"][0]["f"])
        levels = f0 * np.array([0.025, 0.08, 0.18, 0.35, 0.60, 1.0])
        ax.contour(xx, yy, zz, levels=levels, colors=LIGHT, linestyles="--", linewidths=0.8)
        _trajectory(ax, run, BLUE, np.zeros(2), annotate_first=gamma != 1)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        kappa = max(gamma, 1.0 / gamma)
        update_word = "update" if _updates(run) == 1 else "updates"
        ax.set_title(
            rf"({chr(97+i)}) $\gamma={gamma:g}$, $\kappa={kappa:g}$"
            + f"\n{_updates(run)} {update_word} to the gradient threshold",
            pad=11,
        )
    fig.suptitle("Quadratic problem: level sets and numerical iterates", fontsize=14)
    return _save(fig, output, "01_quadratic_paths")


def _quadratic_convergence(runs: list[dict], output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), layout="constrained")
    pairs = [
        (1.0, None, BLUE, r"$\gamma=1$ ($\kappa=1$)"),
        (3.0, 1.0 / 3.0, GREEN, r"$\gamma=1/3,3$ ($\kappa=3$)"),
        (10.0, 0.1, ORANGE, r"$\gamma=0.1,10$ ($\kappa=10$)"),
        (100.0, 0.01, PURPLE, r"$\gamma=0.01,100$ ($\kappa=100$)"),
    ]
    kmax = max(_updates(run) for run in runs if run["problem"] == "quadratic")
    for ax in axes:
        for gamma, reciprocal, color, label in pairs:
            run = _lookup(runs, "quadratic", gamma=gamma)
            ks = _values(run, "k")
            rho = ((gamma - 1.0) / (gamma + 1.0)) ** 2
            theoretical = np.ones_like(ks) if gamma == 1 else rho**ks
            if gamma == 1:
                theoretical[1:] = 0.0
            positive = theoretical > 0.0
            ax.plot(ks[positive], theoretical[positive], color=color, linewidth=1.35, label=label)
            measured = _values(run, "relative_gap")
            good = measured > 0.0
            stride = max(1, len(ks) // 19)
            ax.plot(
                ks[good][::stride], measured[good][::stride], linestyle="none", marker="o",
                markersize=3.5, markerfacecolor="white", markeredgewidth=0.8, color=color,
            )
            if reciprocal is not None:
                smaller = _lookup(runs, "quadratic", gamma=reciprocal)
                kk = _values(smaller, "k")
                rr = _values(smaller, "relative_gap")
                stride2 = max(1, len(kk) // 14)
                offset = max(0, stride2 // 2)
                good2 = rr > 0.0
                ax.plot(
                    kk[good2][offset::stride2], rr[good2][offset::stride2], linestyle="none",
                    marker="x", markersize=4.1, markeredgewidth=0.9, color=color,
                )
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1.6)
        ax.axhline(1e-8, color=GRAY, linewidth=1.0, linestyle=(0, (5, 3)), zorder=1)
        ax.set_xlabel("Update index $k$")
        ax.set_ylabel(r"Relative objective gap $R_k=f(x^{(k)})/f(x^{(0)})$")
        _clean_axes(ax)
    axes[0].set_xlim(0, kmax * 1.02)
    axes[1].set_xlim(0, 90)
    axes[0].set_title("(a) Full trajectories")
    axes[1].set_title("(b) Early iterations")
    axes[0].text(
        0.98, 0.62, r"Diagnostic level: $R_k=10^{-8}$", transform=axes[0].transAxes,
        ha="right", fontsize=9, color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    axes[1].legend(loc="upper right")
    axes[1].annotate(
        r"$\gamma=1$: $R_1=0$", xy=(1, 0.0), xycoords=("data", "axes fraction"),
        xytext=(12, 0.12), textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": BLUE},
        color=BLUE, fontsize=9.5,
    )
    axes[0].text(
        0.03, 0.045, "Lines: closed form. Markers: numerical iterates.\n"
        r"Reciprocal $\gamma$ pairs have the same normalized curve.",
        transform=axes[0].transAxes, fontsize=9, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#E1E6EC", "alpha": 0.96, "pad": 5},
    )
    fig.suptitle("Quadratic problem: geometric convergence and conditioning", fontsize=14)
    return _save(fig, output, "02_quadratic_convergence")


def _objective_comparison(
    selected: list[dict], labels: list[str], title: str, stem: str, output: Path
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout="constrained")
    styles = [_nonquadratic_style(run) for run in selected]
    for run, label, style in zip(selected, labels, styles):
        _line(axes[0], run, "f", style)
        _line(axes[1], run, "gap", style, label, log=True)
    for ax in axes:
        ax.set_xlabel("Update index $k$")
        ax.set_xlim(left=0)
        _clean_axes(ax)
    axes[0].axhline(F_STAR, color=GRAY, linestyle=(0, (5, 3)), linewidth=0.9)
    axes[0].set_ylabel(r"Objective value $f(x^{(k)})$")
    axes[0].set_title("(a) Objective value")
    axes[0].set_ylim(0, max(float(r["history"][0]["f"]) for r in selected) * 1.07)
    axes[0].annotate(
        rf"$f^\star={F_STAR:.6f}$", xy=(0.98, F_STAR), xycoords=("axes fraction", "data"),
        xytext=(0, 7), textcoords="offset points", ha="right", fontsize=9.2, color=GRAY,
    )
    axes[1].set_ylabel(r"Objective gap $f(x^{(k)})-f^\star$")
    axes[1].set_title("(b) Suboptimality on a logarithmic scale")
    axes[1].legend(loc="upper right")

    # 原始目标值保留全程；嵌图展示第一次更新之后的小幅变化。
    inset = axes[0].inset_axes([0.44, 0.40, 0.52, 0.50])
    maximum = max(float(r["history"][1]["f"]) for r in selected if len(r["history"]) > 1)
    max_updates = max(_updates(r) for r in selected)
    for run, style in zip(selected, styles):
        _line(inset, run, "f", style, markers=False)
    inset.axhline(F_STAR, color=GRAY, linestyle=(0, (5, 3)), linewidth=0.7)
    inset.set_xlim(1, min(20, max_updates))
    inset.set_ylim(F_STAR - 0.04 * max(1.0, maximum - F_STAR), maximum * 1.025)
    inset.set_title("After the first update", fontsize=9, pad=4)
    inset.tick_params(labelsize=8, length=2)
    _clean_axes(inset)
    fig.suptitle(title, fontsize=14)
    return _save(fig, output, stem)


def _nonquadratic_baseline(runs: list[dict], output: Path) -> list[str]:
    exact = _find_exact(runs)
    baseline = _lookup(runs, "exponential", alpha=0.1, beta=0.7)
    labels = [
        f"Exact search ({_updates(exact)} updates)",
        rf"Backtracking $\alpha=0.1,\ \beta=0.7$ ({_updates(baseline)} updates)",
    ]
    return _objective_comparison(
        [exact, baseline], labels,
        "Exponential-sum problem: exact search and the lecture baseline",
        "03_nonquadratic_baseline", output,
    )


def _nonquadratic_alpha(runs: list[dict], output: Path) -> list[str]:
    selected = [_lookup(runs, "exponential", alpha=a, beta=0.7) for a in [0.01, 0.1, 0.3]]
    labels = [rf"$\alpha={run['alpha']:g}$ ({_updates(run)} updates)" for run in selected]
    return _objective_comparison(
        selected, labels, r"Varying the sufficient-decrease parameter $\alpha$ ($\beta=0.7$)",
        "04_nonquadratic_alpha", output,
    )


def _nonquadratic_beta(runs: list[dict], output: Path) -> list[str]:
    selected = [_lookup(runs, "exponential", alpha=0.1, beta=b) for b in [0.3, 0.5, 0.7, 0.9]]
    labels = [rf"$\beta={run['beta']:g}$ ({_updates(run)} updates)" for run in selected]
    return _objective_comparison(
        selected, labels, r"Varying the step-reduction factor $\beta$ ($\alpha=0.1$)",
        "05_nonquadratic_beta", output,
    )


def _nonquadratic_paths(runs: list[dict], output: Path) -> list[str]:
    selected = [
        _find_exact(runs),
        _lookup(runs, "exponential", alpha=0.1, beta=0.7),
        _lookup(runs, "exponential", alpha=0.1, beta=0.3),
    ]
    titles = ["Exact search", r"Backtracking: $\alpha=0.1,\ \beta=0.7$", r"Backtracking: $\alpha=0.1,\ \beta=0.3$"]
    points = np.concatenate([
        np.column_stack((_values(run, "x1"), _values(run, "x2"))) for run in selected
    ])
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    span = float(np.max(upper - lower)) * 1.32
    xlim = (center[0] - span / 2, center[0] + span / 2)
    ylim = (center[1] - span / 2, center[1] + span / 2)
    xx, yy = np.meshgrid(np.linspace(*xlim, 440), np.linspace(*ylim, 440))
    zz = np.exp(xx + 3 * yy - 0.1) + np.exp(xx - 3 * yy - 0.1) + np.exp(-xx - 0.1)
    levels = F_STAR + np.array([0.04, 0.12, 0.35, 0.9, 2.2, 5, 11, 25, 55])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7), layout="constrained")
    for i, (ax, run, title, style) in enumerate(zip(axes, selected, titles, STYLES)):
        ax.contour(xx, yy, zz, levels=levels, colors=LIGHT, linestyles="--", linewidths=0.75)
        _trajectory(ax, run, style["color"], X_STAR, annotate_first=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"({chr(97+i)}) {title}\n{_updates(run)} updates", pad=11)
    fig.suptitle("Exponential-sum problem: paths from the same initial point", fontsize=14)
    return _save(fig, output, "06_nonquadratic_paths")


def _nonquadratic_steps(runs: list[dict], output: Path) -> list[str]:
    selected = [_lookup(runs, "exponential", alpha=a, beta=0.7) for a in [0.01, 0.1, 0.3]]
    slow = selected[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout="constrained")
    for run in selected:
        style = _nonquadratic_style(run)
        rows = [r for r in run["history"] if r.get("step") is not None]
        kk = np.asarray([r["k"] for r in rows], float)
        steps = np.asarray([r["step"] for r in rows], float)
        axes[0].step(kk, steps, where="post", color=style["color"], linestyle=style["linestyle"],
                     label=rf"$\alpha={run['alpha']:g}$", linewidth=1.4)
        axes[0].plot(kk, steps, linestyle="none", color=style["color"], marker=style["marker"],
                     markersize=2.8, markerfacecolor="white", markevery=max(1, len(kk)//24))
    maximum_updates = max(_updates(r) for r in selected)
    axes[0].set_xlim(0, min(50, maximum_updates))
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel("Update index $k$")
    axes[0].set_ylabel(r"Accepted step length $t_k$")
    axes[0].set_title("(a) Accepted steps (first 50 updates)")
    axes[0].legend(loc="upper right")

    gaps = _values(slow, "gap")
    ks = _values(slow, "k")
    ratios = np.divide(gaps[1:], gaps[:-1], out=np.full_like(gaps[1:], np.nan), where=gaps[:-1] > 0)
    good = np.isfinite(ratios)
    axes[1].plot(ks[:-1][good], ratios[good], color=PURPLE, linewidth=1.5, label="Measured gap ratio",
                 marker="o", markersize=3, markerfacecolor="white", markevery=max(1, len(ratios)//20))
    steps = np.asarray([r["step"] for r in slow["history"] if r.get("step") is not None], float)
    tail_steps = steps[-min(20, len(steps)):]
    tail_step = float(np.median(tail_steps))
    if not np.allclose(tail_steps, tail_step, rtol=1e-12, atol=1e-14):
        raise ValueError("The selected slow run has no constant late step; the single-step local prediction is inapplicable.")
    local_largest_eigenvalue = 4.5 * F_STAR
    predicted_ratio = (1.0 - tail_step * local_largest_eigenvalue) ** 2
    axes[1].axhline(predicted_ratio, color=ORANGE, linestyle="--", linewidth=1.25,
                    label=rf"Local prediction: $(1-t\lambda_2^\star)^2={predicted_ratio:.6f}$")
    axes[1].set_xlabel("Update index $k$")
    axes[1].set_ylabel(r"Gap ratio $(f_{k+1}-f^\star)/(f_k-f^\star)$")
    axes[1].set_title(r"(b) Gap ratios: $\alpha=0.01,\ \beta=0.7$")
    axes[1].set_ylim(0, max(1.0, min(1.2, float(np.nanmax(ratios[good])) * 1.05)))
    axes[1].legend(loc="lower right", fontsize=8.8)
    power = int(round(np.log(tail_step) / np.log(0.7)))
    axes[1].text(
        0.12, 0.50, rf"Late step: $t=0.7^{{{power}}}={tail_step:.5f}$" + "\n"
        r"$\lambda_2^\star=\lambda_{\max}(H(x^\star))$. Local approximation.",
        transform=axes[1].transAxes, fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 4},
    )
    for ax in axes:
        _clean_axes(ax)
    fig.suptitle(r"Accepted steps and the late quadratic approximation ($\beta=0.7$)", fontsize=14)
    return _save(fig, output, "07_nonquadratic_steps")


def _first_line_search(runs: list[dict], output: Path) -> list[str]:
    exact = _find_exact(runs)
    baseline = _lookup(runs, "exponential", alpha=0.1, beta=0.7)
    point = np.array([exact["history"][0]["x1"], exact["history"][0]["x2"]], dtype=float)
    baseline_point = np.array([baseline["history"][0]["x1"], baseline["history"][0]["x2"]])
    if not np.array_equal(point, baseline_point):
        raise ValueError("The line-search illustration requires the same initial point.")
    a = np.exp(point[0] + 3 * point[1] - 0.1)
    b = np.exp(point[0] - 3 * point[1] - 0.1)
    c = np.exp(-point[0] - 0.1)
    gradient = np.array([a + b - c, 3 * (a - b)])
    initial_f = float(a + b + c)
    grad_sq = float(gradient @ gradient)
    t_exact = float(exact["history"][0]["step"])
    t_backtracking = float(baseline["history"][0]["step"])
    t_previous = t_backtracking / float(baseline["beta"])
    right = max(t_previous, t_exact) * 1.10
    ts = np.linspace(0, right, 900)
    along = point[:, None] - gradient[:, None] * ts[None, :]
    aa = np.exp(along[0] + 3 * along[1] - 0.1)
    bb = np.exp(along[0] - 3 * along[1] - 0.1)
    cc = np.exp(-along[0] - 0.1)
    phi = aa + bb + cc
    derivative = -gradient[0] * (aa + bb - cc) - gradient[1] * 3 * (aa - bb)
    armijo = initial_f - float(baseline["alpha"]) * ts * grad_sq

    def along_value(t: float) -> float:
        z = point - t * gradient
        return float(np.exp(z[0] + 3 * z[1] - 0.1) + np.exp(z[0] - 3 * z[1] - 0.1) + np.exp(-z[0] - 0.1))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), layout="constrained")
    axes[0].plot(ts, phi, color=BLUE, linewidth=1.7, label=r"$\phi(t)=f(x^{(0)}-t g_0)$")
    axes[0].plot(ts, armijo, color=ORANGE, linestyle="--", linewidth=1.4,
                 label=r"Armijo line: $f_0-0.1t\Vert g_0\Vert^2$")
    axes[0].scatter(t_exact, along_value(t_exact), marker="*", s=115, color=BLUE, edgecolor="white",
                    linewidth=0.6, zorder=6, label="Exact minimum")
    axes[0].scatter(t_backtracking, along_value(t_backtracking), s=42, color=GREEN, edgecolor="white",
                    linewidth=0.6, zorder=6, label="Accepted backtracking candidate")
    axes[0].scatter(t_previous, along_value(t_previous), s=38, marker="x", color=GRAY,
                    linewidth=1.1, zorder=6, label="Previous, rejected candidate")
    axes[0].axvline(t_exact, color=BLUE, linewidth=0.7, linestyle=":", alpha=0.8)
    axes[0].set_xlabel("Step length $t$ along the fixed direction $-g_0$")
    axes[0].set_ylabel(r"One-dimensional objective $\phi(t)$")
    axes[0].set_title("(a) Minimize the function along one ray")
    axes[0].set_ylim(0, max(initial_f, float(phi[-1])) * 1.05)
    axes[0].legend(loc="upper center", fontsize=8.5)
    axes[1].plot(ts, derivative, color=BLUE, linewidth=1.6)
    axes[1].axhline(0, color=GRAY, linewidth=0.9)
    axes[1].axvline(t_exact, color=ORANGE, linestyle="--", linewidth=1.3)
    axes[1].scatter(t_exact, 0, color=ORANGE, s=38, zorder=5)
    axes[1].text(
        0.04, 0.95, rf"$t^\star={t_exact:.8f}$" + "\n" + r"$\phi'(t^\star)=0$ and $\phi''(t)>0$",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#DDE5EC", "alpha": 0.96, "pad": 5},
    )
    axes[1].set_xlabel("Step length $t$ along the fixed direction $-g_0$")
    axes[1].set_ylabel(r"Directional derivative $\phi'(t)$")
    axes[1].set_title("(b) The same minimizer is the derivative's zero")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.set_xlim(0, right)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        _clean_axes(ax)
    fig.suptitle(r"One fixed iterate, one search direction: $x^{(0)}=(1,1)$", fontsize=14)
    return _save(fig, output, "08_first_line_search")


def generate_figures(data_path: str | Path, output_dir: str | Path) -> list[str]:
    """返回全部 PDF 与 PNG 的绝对路径；不更改任何实验数据。"""
    data_path, output = Path(data_path), Path(output_dir)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    runs = data["runs"]
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with plt.rc_context(_style()):
        for draw in [
            _quadratic_paths,
            _quadratic_convergence,
            _nonquadratic_baseline,
            _nonquadratic_alpha,
            _nonquadratic_beta,
            _nonquadratic_paths,
            _nonquadratic_steps,
            _first_line_search,
        ]:
            paths.extend(draw(runs, output))
    return paths


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "results" / "experiment_data.json")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args(argv)
    for path in generate_figures(args.data, args.output):
        print(path)


if __name__ == "__main__":
    main()
