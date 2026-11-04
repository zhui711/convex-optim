"""凸优化第一次大作业：两个二维问题的可复现梯度下降实验。"""
from .problems import Quadratic, ExponentialSum
from .line_search import SearchResult, quadratic_exact, exact_bisection, armijo_backtracking
from .optimize import OptimizationResult, gradient_descent

__all__ = ["Quadratic", "ExponentialSum", "SearchResult", "OptimizationResult",
           "quadratic_exact", "exact_bisection", "armijo_backtracking", "gradient_descent"]
