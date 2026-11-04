"""作业中的两个目标函数，以及独立于求解器的解析诊断量。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

Array = np.ndarray


def _point(x: Array) -> Array:
    p = np.asarray(x, dtype=float)
    if p.shape != (2,):
        raise ValueError("本作业的点必须是两个分量的一维向量。")
    return p


class SmoothProblem(Protocol):
    name: str
    x_star: Array
    f_star: float

    def value(self, x: Array) -> float: ...
    def gradient(self, x: Array) -> Array: ...
    def hessian(self, x: Array) -> Array: ...
    def gap(self, x: Array) -> float: ...
    def value_difference(self, x: Array, displacement: Array) -> float: ...
    def directional_derivative(self, x: Array, direction: Array) -> float: ...


@dataclass(frozen=True)
class Quadratic:
    """f(x)=(x1^2+gamma*x2^2)/2，gamma>0。"""
    gamma: float
    name: str = "quadratic"
    f_star: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma 必须为正的有限数。")

    @property
    def x_star(self) -> Array:
        return np.zeros(2)

    @property
    def x0(self) -> Array:
        return np.array([self.gamma, 1.0])

    @property
    def q(self) -> float:
        """有符号坐标因子 q=(gamma-1)/(gamma+1)。"""
        return (self.gamma - 1.0) / (self.gamma + 1.0)

    @property
    def rho(self) -> float:
        """目标差的每步收缩因子 rho=q^2，R_k=rho^k。"""
        return self.q ** 2

    @property
    def coordinate_contraction(self) -> float:
        return abs(self.q)

    @property
    def condition_number(self) -> float:
        return max(self.gamma, 1.0 / self.gamma)

    def value(self, x: Array) -> float:
        p = _point(x)
        return float(0.5 * (p[0] ** 2 + self.gamma * p[1] ** 2))

    def gradient(self, x: Array) -> Array:
        p = _point(x)
        return np.array([p[0], self.gamma * p[1]])

    def hessian(self, x: Array) -> Array:
        _point(x)
        return np.diag([1.0, self.gamma])

    def gap(self, x: Array) -> float:
        return self.value(x)

    def value_difference(self, x: Array, displacement: Array) -> float:
        """精确恒等式 g(x)^T s+s^T H s/2，避免两个近似相等数相减。"""
        p, s = _point(x), _point(displacement)
        return float(p[0] * s[0] + self.gamma * p[1] * s[1]
                     + 0.5 * (s[0] ** 2 + self.gamma * s[1] ** 2))

    def directional_derivative(self, x: Array, direction: Array) -> float:
        return float(np.dot(self.gradient(x), _point(direction)))

    def closed_form(self, k: int) -> Array:
        """只适用于题设初值 (gamma,1) 与精确直线搜索。"""
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("k 必须为非负整数。")
        return (self.q ** k) * np.array([self.gamma, (-1.0) ** k])

    def predicted_iterations(self, grad_tol: float = 1e-6) -> int:
        """由 ||g_k||=sqrt(2)*gamma*|q|^k 计算首次达标次数。"""
        if not math.isfinite(grad_tol) or grad_tol <= 0:
            raise ValueError("grad_tol 必须为正的有限数。")
        initial = math.sqrt(2.0) * self.gamma
        if initial <= grad_tol:
            return 0
        if self.q == 0:
            return 1
        k = max(0, math.ceil(math.log(grad_tol / initial) / math.log(abs(self.q))))
        # 修正对数运算在整数边界附近的舍入；不依赖实验结果。
        while initial * abs(self.q) ** k > grad_tol:
            k += 1
        while k > 0 and initial * abs(self.q) ** (k - 1) <= grad_tol:
            k -= 1
        return k

    def relative_gap_iterations(self, tolerance: float = 1e-8) -> int:
        """由 R_k=rho^k 计算诊断阈值的首次到达次数。"""
        if not math.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("relative-gap tolerance 必须在 (0,1) 内。")
        if self.rho == 0:
            return 1
        k = max(0, math.ceil(math.log(tolerance) / math.log(self.rho)))
        while self.rho ** k > tolerance:
            k += 1
        while k > 0 and self.rho ** (k - 1) <= tolerance:
            k -= 1
        return k


class ExponentialSum:
    """f=e^(x1+3x2-.1)+e^(x1-3x2-.1)+e^(-x1-.1)。"""
    name = "exponential"
    f_star = 2.0 * math.sqrt(2.0) * math.exp(-0.1)
    _A = np.array([[1.0, 3.0], [1.0, -3.0], [-1.0, 0.0]])

    @property
    def x_star(self) -> Array:
        return np.array([-0.5 * math.log(2.0), 0.0])

    def _terms(self, x: Array) -> Array:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(self._A @ _point(x) - 0.1)

    def value(self, x: Array) -> float:
        return float(np.sum(self._terms(x)))

    def gradient(self, x: Array) -> Array:
        # u=x1-x1*、v=3*x2。下式与 (a+b-c,3(a-b)) 严格等价，
        # 但在最优点附近不需要对量级约为 1 的两个数做相减。
        p = _point(x)
        u, v = p[0] + 0.5 * math.log(2.0), 3.0 * p[1]
        with np.errstate(over="ignore", invalid="ignore"):
            return np.array([
                self.f_star * (np.sinh(u) + np.exp(u) * np.sinh(v / 2.0) ** 2),
                1.5 * self.f_star * np.exp(u) * np.sinh(v),
            ])

    def hessian(self, x: Array) -> Array:
        a, b, c = self._terms(x)
        return np.array([[a + b + c, 3.0 * (a - b)],
                         [3.0 * (a - b), 9.0 * (a + b)]])

    def gap(self, x: Array) -> float:
        """稳定地计算 f-f*；只用于诊断，不参与外层停止判断。"""
        p = _point(x)
        u, v = p[0] + 0.5 * math.log(2.0), 3.0 * p[1]
        with np.errstate(over="ignore", invalid="ignore"):
            return float(self.f_star * (2.0 * np.sinh(u / 2.0) ** 2
                         + np.exp(u) * np.sinh(v / 2.0) ** 2))

    def value_difference(self, x: Array, displacement: Array) -> float:
        """Δf=Σ exp(a_i^T x-.1)*expm1(a_i^T s)，精确的函数差恒等式。"""
        with np.errstate(over="ignore", invalid="ignore"):
            changes = self._terms(x) * np.expm1(self._A @ _point(displacement))
        if np.all(np.isfinite(changes)):
            return math.fsum(float(c) for c in changes)
        return float(np.sum(changes))

    def directional_derivative(self, x: Array, direction: Array) -> float:
        p, d = _point(x), _point(direction)
        with np.errstate(over="ignore", invalid="ignore"):
            slope = float(np.dot(self.gradient(p), d))
        if math.isfinite(slope):
            return slope
        # 很远的试探点可能指数溢出。按最大指数缩放仅确定符号，
        # 保持二分法的左右区间判断；有限范围内仍返回真实方向导数。
        z = self._A @ p - 0.1
        rates = self._A @ d
        shifted = np.exp(z - np.max(z))
        scaled = math.fsum(float(r * w) for r, w in zip(rates, shifted))
        return math.copysign(math.inf, scaled) if scaled else 0.0
