"""统一的梯度下降外层循环与可审计的迭代日志。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .problems import Quadratic, SmoothProblem, _point
from .line_search import quadratic_exact, exact_bisection, armijo_backtracking


@dataclass
class OptimizationResult:
    history: list[dict]
    summary: dict

    @property
    def x(self) -> np.ndarray:
        return np.array([self.summary["final_x1"], self.summary["final_x2"]])


def gradient_descent(problem: SmoothProblem, x0, method="exact", *,
                     grad_tol=1e-6, max_updates=10000, alpha=0.1, beta=0.7,
                     exact_xtol=1e-12, exact_rtol=1e-12) -> OptimizationResult:
    """x_(k+1)=x_k-t_k*grad f(x_k)，不归一化梯度。

    唯一收敛停止条件为 ||grad f(x_k)||_2 <= grad_tol。
    gap/最优点距离用于诊断；问题模块中中心化的稳定导数与原式等价。
    k 计已接受的更新次数；每行 step
    指从该行出发的步长，最后一行为 None。累计成本 n_* 是到达该行
    并计算其梯度后的成本，不包含该行尚未执行的 outgoing search。
    n_f 包含完整目标值和稳定函数差查询；n_delta 是其子计数，
    n_full_f=n_f-n_delta。n_g 仅计外层梯度，n_slope 单独计内层方向导数。
    gap/point_error 等解析诊断不计入求解器 oracle 成本。
    """
    if not math.isfinite(grad_tol) or grad_tol <= 0:
        raise ValueError("grad_tol 必须为正的有限数。")
    if not isinstance(max_updates, int) or max_updates < 0:
        raise ValueError("max_updates 必须为非负整数。")
    if method not in {"quadratic_exact", "exact", "exact_bisection", "backtracking",
                      "armijo", "armijo_backtracking"}:
        raise ValueError("仅支持精确直线搜索或 Armijo 回溯的梯度下降。")
    if method == "quadratic_exact" and not isinstance(problem, Quadratic):
        raise TypeError("quadratic_exact 仅用于二次题。")
    x = _point(x0).copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("初值必须有限。")
    f = problem.value(x)
    if not math.isfinite(f):
        raise ValueError("初始目标值溢出或无效。")
    initial_gap = problem.gap(x)
    n_f, n_g, n_slope, n_delta = 1, 0, 0, 0
    history = []
    status = "max_updates"
    total_trials = total_inner = total_expansions = 0

    for k in range(max_updates + 1):
        g = problem.gradient(x)
        n_g += 1
        grad_norm = float(np.linalg.norm(g))
        gap = problem.gap(x)
        row = {
            "k": k, "x1": float(x[0]), "x2": float(x[1]), "f": float(f),
            "gap": float(gap),
            "relative_gap": float(gap / initial_gap) if initial_gap > 0 else 0.0,
            "grad_norm": grad_norm, "step": None,
            "next_search_trials": 0, "next_search_value_evals": 0,
            "next_search_difference_evals": 0, "next_search_slope_evals": 0,
            "next_search_inner_iterations": 0, "next_search_bracket_expansions": 0,
            "normalized_decrease": None, "prev_rejected_normalized_decrease": None,
            "armijo_margin": None, "slope_at_step": None, "search_bracket_width": None,
            "n_f": n_f, "n_g": n_g, "n_slope": n_slope, "n_delta": n_delta,
            "n_full_f": n_f - n_delta,
        }
        history.append(row)
        if not math.isfinite(grad_norm):
            status = "nonfinite_gradient"
            break
        if grad_norm <= grad_tol:
            status = "converged"
            break
        if k == max_updates:
            break
        if method == "quadratic_exact" or (method == "exact" and isinstance(problem, Quadratic)):
            search = quadratic_exact(problem, x, g)
        elif method in {"exact", "exact_bisection"}:
            search = exact_bisection(problem, x, g, xtol=exact_xtol, rtol=exact_rtol)
        else:
            search = armijo_backtracking(problem, x, g, alpha=alpha, beta=beta)
        row.update({
            "step": search.step, "next_search_trials": search.trials,
            "next_search_value_evals": search.value_evals,
            "next_search_difference_evals": search.difference_evals,
            "next_search_slope_evals": search.slope_evals,
            "next_search_inner_iterations": search.inner_iterations,
            "next_search_bracket_expansions": search.bracket_expansions,
            "normalized_decrease": search.normalized_decrease,
            "prev_rejected_normalized_decrease": search.prev_rejected_normalized_decrease,
            "armijo_margin": search.armijo_margin, "slope_at_step": search.slope_at_step,
            "search_bracket_width": search.bracket_width,
        })
        n_f += search.value_evals
        n_delta += search.difference_evals
        n_slope += search.slope_evals
        total_trials += search.trials
        total_inner += search.inner_iterations
        total_expansions += search.bracket_expansions
        x = x - search.step * g
        f = search.value  # 复用搜索已计算的目标值，避免重复计数。

    last = history[-1]
    summary = {
        "problem": problem.name, "method": method, "status": status,
        "iterations": last["k"], "grad_tol": float(grad_tol),
        "final_x1": last["x1"], "final_x2": last["x2"],
        "final_f": last["f"], "final_gap": last["gap"],
        "final_relative_gap": last["relative_gap"],
        "final_grad_norm": last["grad_norm"],
        "point_error": float(np.linalg.norm(x - problem.x_star)),
        "n_f": n_f, "n_g": n_g, "n_slope": n_slope,
        "n_delta": n_delta, "n_full_f": n_f - n_delta,
        "search_trials": total_trials, "inner_iterations": total_inner,
        "bracket_expansions": total_expansions,
        "initial_x1": float(_point(x0)[0]), "initial_x2": float(_point(x0)[1]),
        "initial_f": history[0]["f"], "initial_gap": float(initial_gap),
    }
    return OptimizationResult(history=history, summary=summary)
