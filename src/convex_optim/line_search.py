"""沿未归一化负梯度的直线搜索。内层一维搜索不会改变外层算法。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .problems import Quadratic, SmoothProblem, _point


@dataclass(frozen=True)
class SearchResult:
    step: float
    value: float
    value_evals: int
    slope_evals: int
    trials: int
    inner_iterations: int
    bracket_expansions: int
    difference_evals: int = 0
    normalized_decrease: float | None = None
    prev_rejected_normalized_decrease: float | None = None
    armijo_margin: float | None = None
    slope_at_step: float | None = None
    bracket_width: float | None = None

    @property
    def full_value_evals(self) -> int:
        return self.value_evals - self.difference_evals


# 计数约定：value_evals 包括完整 f 与稳定 Δf 的标量查询；
# difference_evals 是其中的 Δf 子计数，因此两者不能相加。
# slope_evals 单独计方向导数，不重复计入外层梯度调用。


def _validate_direction(x, gradient):
    p, g = _point(x), _point(gradient)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
        raise ValueError("直线搜索起点与梯度必须有限。")
    return p, g, float(np.dot(g, g))


def _finish(problem, x, g, step, **kwargs) -> SearchResult:
    """精确搜索后计算接受点目标与稳定下降比；二者均如实计数。"""
    delta = problem.value_difference(x, -step * g)
    value = problem.value(x - step * g)
    norm2 = float(np.dot(g, g))
    ratio = -delta / (step * norm2) if step > 0 and norm2 > 0 else None
    return SearchResult(step=step, value=value, value_evals=2,
                        difference_evals=1, normalized_decrease=ratio, **kwargs)


def quadratic_exact(problem: Quadratic, x, gradient) -> SearchResult:
    """二次函数解析步长 t=(g^Tg)/(g^THg)。"""
    p, g, norm2 = _validate_direction(x, gradient)
    if not isinstance(problem, Quadratic):
        raise TypeError("quadratic_exact 仅用于 Quadratic。")
    step = norm2 / float(g @ problem.hessian(p) @ g) if norm2 > 0 else 0.0
    return _finish(problem, p, g, step, slope_evals=0, trials=1,
                   inner_iterations=0, bracket_expansions=0, bracket_width=0.0)


def exact_bisection(problem: SmoothProblem, x, gradient, *, initial_step=1.0,
                    xtol=1e-12, rtol=1e-12, max_iterations=200,
                    max_expansions=60) -> SearchResult:
    """数值实现 t*=argmin_{t>=0} f(x-tg)。

    令 phi(t)=f(x-tg)。phi'(0)=-||g||^2<0，严格凸问题中
    phi' 严格递增。先倍增右端点夹住零点，再二分 phi'(t)=0。
    xtol/rtol 只决定内层步长精度，不是外层优化的停止阈值。
    """
    p, g, norm2 = _validate_direction(x, gradient)
    if initial_step <= 0 or not math.isfinite(initial_step):
        raise ValueError("initial_step 必须为正的有限数。")
    if xtol <= 0 or rtol < 0 or not math.isfinite(xtol + rtol):
        raise ValueError("xtol>0、rtol>=0 且必须有限。")
    if max_iterations < 1 or max_expansions < 0:
        raise ValueError("搜索迭代上限不合法。")
    if norm2 == 0:
        return _finish(problem, p, g, 0.0, slope_evals=0, trials=0,
                       inner_iterations=0, bracket_expansions=0, slope_at_step=0.0,
                       bracket_width=0.0)
    d = -g
    slope_evals = 0

    def slope(t):
        nonlocal slope_evals
        slope_evals += 1
        result = problem.directional_derivative(p + t * d, d)
        if math.isnan(result):
            raise ArithmeticError("方向导数为 NaN，无法判定二分搜索区间。")
        return result

    low, high = 0.0, float(initial_step)
    high_slope = slope(high)
    expansions = 0
    while high_slope < 0:
        if expansions >= max_expansions:
            raise RuntimeError("精确直线搜索未在允许的倍增次数内夹住零点。")
        low, high = high, 2.0 * high
        expansions += 1
        high_slope = slope(high)
    if high_slope == 0:
        return _finish(problem, p, g, high, slope_evals=slope_evals,
                       trials=slope_evals, inner_iterations=0,
                       bracket_expansions=expansions, slope_at_step=0.0,
                       bracket_width=0.0)
    for inner in range(1, max_iterations + 1):
        middle = 0.5 * (low + high)
        middle_slope = slope(middle)
        if middle_slope == 0:
            low = high = middle
            break
        if middle_slope < 0:
            low = middle
        else:
            high = middle
        if high - low <= xtol + rtol * max(abs(low), abs(high)):
            middle = 0.5 * (low + high)
            middle_slope = slope(middle)
            break
    else:
        raise RuntimeError("精确直线搜索二分迭代达到上限。")
    return _finish(problem, p, g, middle, slope_evals=slope_evals,
                   trials=slope_evals, inner_iterations=inner,
                   bracket_expansions=expansions, slope_at_step=middle_slope,
                   bracket_width=high - low)


def armijo_backtracking(problem: SmoothProblem, x, gradient, *, alpha=0.1,
                        beta=0.7, initial_step=1.0, max_trials=1000) -> SearchResult:
    """从 t=1 开始乘 beta，直到 Δf <= -alpha*t*||g||^2。"""
    p, g, norm2 = _validate_direction(x, gradient)
    if not 0 < alpha < 0.5 or not 0 < beta < 1:
        raise ValueError("按课件约定，alpha∈(0,0.5)、beta∈(0,1)。")
    if initial_step <= 0 or not math.isfinite(initial_step) or max_trials < 1:
        raise ValueError("初始步长或最大试探次数不合法。")
    t = float(initial_step)
    prev_ratio = None
    for trial in range(1, max_trials + 1):
        # 用严格等价的 Δf 形式测试 Armijo，避免末期 f(x+t*d)-f(x) 相消。
        delta = problem.value_difference(p, -t * g)
        rhs_change = -alpha * t * norm2
        ratio = -delta / (t * norm2) if norm2 > 0 else None
        if math.isfinite(delta) and delta <= rhs_change:
            value = problem.value(p - t * g)
            return SearchResult(
                step=t, value=value, value_evals=trial + 1,
                difference_evals=trial, slope_evals=0, trials=trial,
                inner_iterations=trial - 1, bracket_expansions=0,
                normalized_decrease=ratio,
                prev_rejected_normalized_decrease=prev_ratio,
                armijo_margin=rhs_change - delta,
            )
        prev_ratio = ratio
        t *= beta
        if t == 0:
            raise RuntimeError("回溯步长下溢。")
    raise RuntimeError("回溯搜索达到最大试探次数，尚未满足 Armijo 条件。")
