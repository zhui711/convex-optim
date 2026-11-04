"""将数值实现与独立的数学性质核对；只依赖标准库 unittest 与 NumPy。"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

# 允许直接 python -m unittest discover -s tests -v，不要求先安装项目。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convex_optim import (Quadratic, ExponentialSum, gradient_descent,
                          quadratic_exact, exact_bisection, armijo_backtracking)


GAMMAS = [0.01, 0.1, 1.0 / 3.0, 1.0, 3.0, 10.0, 100.0]


class QuadraticTests(unittest.TestCase):
    def test_closed_form_entire_trajectory(self):
        for gamma in GAMMAS:
            with self.subTest(gamma=gamma):
                p = Quadratic(gamma)
                result = gradient_descent(p, [gamma, 1.0], "quadratic_exact")
                for row in result.history:
                    k = row["k"]
                    q = (gamma - 1.0) / (gamma + 1.0)
                    expected = q ** k * np.array([gamma, (-1.0) ** k])
                    np.testing.assert_allclose([row["x1"], row["x2"]], expected,
                                               rtol=5e-10, atol=1e-12)
                    self.assertAlmostEqual(row["relative_gap"], q ** (2 * k), places=12)
                    if row["step"] is not None:
                        self.assertAlmostEqual(row["step"], 2.0 / (1.0 + gamma), places=11)
                        self.assertAlmostEqual(row["normalized_decrease"], 0.5, places=12)

    def test_gradient_stopping_first_hit(self):
        # 独立解析基准：sqrt(2)*gamma*|q|^K <= 1e-6 首次成立。
        expected = [478, 60, 19, 1, 23, 83, 939]
        for gamma, count in zip(GAMMAS, expected):
            with self.subTest(gamma=gamma):
                p = Quadratic(gamma)
                r = gradient_descent(p, p.x0, "quadratic_exact")
                self.assertEqual(r.summary["status"], "converged")
                self.assertEqual(r.summary["iterations"], count)
                self.assertEqual(p.predicted_iterations(), count)
                self.assertLessEqual(r.history[-1]["grad_norm"], 1e-6)
                self.assertGreater(r.history[-2]["grad_norm"], 1e-6)

    def test_relative_gap_diagnostic_first_hit(self):
        expected = [461, 46, 14, 1, 14, 46, 461]
        for gamma, count in zip(GAMMAS, expected):
            with self.subTest(gamma=gamma):
                p = Quadratic(gamma)
                r = gradient_descent(p, p.x0, "quadratic_exact")
                first = next(row["k"] for row in r.history if row["relative_gap"] <= 1e-8)
                self.assertEqual(first, count)
                self.assertEqual(p.relative_gap_iterations(), count)
                self.assertEqual(p.condition_number, max(gamma, 1.0 / gamma))

    def test_gamma_one_any_initial_point_one_step(self):
        p = Quadratic(1.0)
        for x0 in [[1.0, 1.0], [2.3, -8.1], [-1.0, 0.0]]:
            r = gradient_descent(p, x0, "quadratic_exact")
            self.assertEqual(r.summary["iterations"], 1)
            np.testing.assert_array_equal(r.x, np.zeros(2))

    def test_analytic_step_minimizes_quadratic_line(self):
        p = Quadratic(7.0)
        x = np.array([2.4, -0.3])
        g = p.gradient(x)
        search = quadratic_exact(p, x, g)
        self.assertLess(abs(g @ p.gradient(x - search.step * g)), 1e-13)
        for dt in [-0.01, 0.01]:
            self.assertGreater(p.value(x - (search.step + dt) * g), search.value)

    def test_reciprocal_parameters_same_normalized_rate(self):
        for gamma in [0.01, 0.1, 1.0 / 3.0]:
            a, b = Quadratic(gamma), Quadratic(1.0 / gamma)
            self.assertAlmostEqual(a.rho, b.rho, places=15)
            self.assertEqual(a.relative_gap_iterations(), b.relative_gap_iterations())


class ExponentialTests(unittest.TestCase):
    def setUp(self):
        self.p = ExponentialSum()

    @staticmethod
    def independent_value(x):
        return math.exp(x[0] + 3 * x[1] - 0.1) + math.exp(x[0] - 3 * x[1] - 0.1) + math.exp(-x[0] - 0.1)

    def test_analytic_optimum_and_positive_hessian(self):
        p = self.p
        np.testing.assert_allclose(p.x_star, [-math.log(2.0) / 2.0, 0.0], atol=0)
        self.assertAlmostEqual(p.f_star, 2.0 * math.sqrt(2.0) * math.exp(-0.1), places=15)
        self.assertAlmostEqual(p.value(p.x_star), p.f_star, places=14)
        np.testing.assert_array_equal(p.gradient(p.x_star), [0.0, 0.0])
        self.assertEqual(p.gap(p.x_star), 0.0)
        for x in [[0, 0], [1, 1], [-2, -0.4], p.x_star]:
            self.assertGreater(np.linalg.eigvalsh(p.hessian(x)).min(), 0.0)
        eigenvalues = np.linalg.eigvalsh(p.hessian(p.x_star))
        self.assertAlmostEqual(eigenvalues[1] / eigenvalues[0], 4.5, places=13)

    def test_gradient_and_hessian_finite_differences(self):
        p = self.p
        h = 1e-5
        for x in [np.array([0.2, -0.3]), np.array([1.0, 1.0]), np.array([-1.2, 0.15])]:
            with self.subTest(x=x):
                fd_gradient = np.array([(self.independent_value(x + h * e) - self.independent_value(x - h * e)) / (2 * h) for e in np.eye(2)])
                fd_hessian = np.column_stack([(p.gradient(x + h * e) - p.gradient(x - h * e)) / (2 * h) for e in np.eye(2)])
                np.testing.assert_allclose(p.gradient(x), fd_gradient, rtol=3e-9, atol=3e-9)
                np.testing.assert_allclose(p.hessian(x), fd_hessian, rtol=3e-9, atol=3e-9)

    def test_stable_gap_matches_function_and_small_distance(self):
        p = self.p
        for x in [[1, 1], [0.2, -0.3], [-1.2, 0.15]]:
            self.assertAlmostEqual(p.gap(x), self.independent_value(x) - p.f_star, places=12)
        displacement = np.array([1e-10, -2e-10])
        x = p.x_star + displacement
        # 在直接相减已接近机器舍入尺度时，平方恒等式仍保留正误差。
        expected_taylor = 0.5 * displacement @ p.hessian(p.x_star) @ displacement
        self.assertGreater(p.gap(x), 0.0)
        np.testing.assert_allclose(p.gap(x), expected_taylor, rtol=1e-7, atol=0)

    def test_stable_difference_identity_and_local_expansion(self):
        p = self.p
        x, s = np.array([0.4, 0.2]), np.array([-0.02, 0.01])
        self.assertAlmostEqual(p.value_difference(x, s), self.independent_value(x + s) - self.independent_value(x), places=13)
        x = p.x_star + np.array([1e-7, -2e-7])
        s = -0.1 * p.gradient(x)
        local = p.gradient(x) @ s + 0.5 * s @ p.hessian(x) @ s
        np.testing.assert_allclose(p.value_difference(x, s), local, rtol=2e-6, atol=1e-22)

    def test_exact_search_root_and_orthogonal_gradients(self):
        p = self.p
        r = gradient_descent(p, [1.0, 1.0], "exact")
        self.assertEqual(r.summary["status"], "converged")
        for before, after in zip(r.history, r.history[1:]):
            g = p.gradient([before["x1"], before["x2"]])
            g_next = p.gradient([after["x1"], after["x2"]])
            cosine = abs(g @ g_next) / (np.linalg.norm(g) * np.linalg.norm(g_next))
            self.assertLess(cosine, 1e-8)
            self.assertLess(after["gap"], before["gap"])
            self.assertLessEqual(before["search_bracket_width"], 1e-12 * (1.0 + before["step"]) + 1e-20)

    def test_exact_search_bracket_expansion(self):
        p = self.p
        x = np.array([0.0, 0.0])
        search = exact_bisection(p, x, p.gradient(x), initial_step=1e-4)
        self.assertGreater(search.bracket_expansions, 0)
        np.testing.assert_allclose(x - search.step * p.gradient(x), p.x_star, atol=1e-11)

    def test_axis_initialization_exact_one_step(self):
        for x1 in [-1.0, 0.0, 1.0]:
            with self.subTest(x1=x1):
                r = gradient_descent(self.p, [x1, 0.0], "exact")
                self.assertEqual(r.summary["iterations"], 1)
                self.assertEqual(r.summary["final_x2"], 0.0)
                self.assertLessEqual(r.summary["final_grad_norm"], 1e-6)

    def test_backtracking_parameter_grid_and_armijo(self):
        p = self.p
        for alpha in [0.01, 0.1, 0.3]:
            for beta in [0.3, 0.5, 0.7, 0.9]:
                with self.subTest(alpha=alpha, beta=beta):
                    r = gradient_descent(p, [1.0, 1.0], "backtracking", alpha=alpha, beta=beta)
                    self.assertEqual(r.summary["status"], "converged")
                    self.assertLessEqual(r.summary["final_grad_norm"], 1e-6)
                    self.assertGreater(r.history[-2]["grad_norm"], 1e-6)
                    for row in r.history[:-1]:
                        self.assertGreaterEqual(row["normalized_decrease"], alpha)
                        self.assertGreaterEqual(row["armijo_margin"], 0.0)
                        self.assertAlmostEqual(row["step"], beta ** (row["next_search_trials"] - 1), places=14)
                        previous = row["prev_rejected_normalized_decrease"]
                        if previous is not None:
                            self.assertLess(previous, alpha)

    def test_different_alpha_can_select_identical_discrete_steps(self):
        a = gradient_descent(self.p, [1, 1], "backtracking", alpha=0.01, beta=0.3)
        b = gradient_descent(self.p, [1, 1], "backtracking", alpha=0.1, beta=0.3)
        np.testing.assert_array_equal(
            [[v["x1"], v["x2"]] for v in a.history],
            [[v["x1"], v["x2"]] for v in b.history])
        for row in a.history[:-1]:
            self.assertGreaterEqual(row["normalized_decrease"], 0.1)
            previous = row["prev_rejected_normalized_decrease"]
            if previous is not None:
                self.assertLess(previous, 0.01)


class LoggingAndValidationTests(unittest.TestCase):
    def test_cost_accounting_and_terminal_row(self):
        for method in ["exact", "backtracking"]:
            r = gradient_descent(ExponentialSum(), [1, 1], method)
            summary = r.summary
            self.assertEqual(summary["n_g"], summary["iterations"] + 1)
            self.assertEqual(summary["n_full_f"], summary["iterations"] + 1)
            self.assertEqual(summary["n_f"], summary["n_delta"] + summary["n_full_f"])
            self.assertEqual(summary["n_f"], 1 + sum(h["next_search_value_evals"] for h in r.history))
            self.assertEqual(summary["n_slope"], sum(h["next_search_slope_evals"] for h in r.history))
            self.assertEqual(summary["n_delta"], sum(h["next_search_difference_evals"] for h in r.history))
            self.assertEqual(summary["search_trials"], sum(h["next_search_trials"] for h in r.history))
            self.assertIsNone(r.history[-1]["step"])
            for key in ["n_f", "n_g", "n_slope", "n_delta", "n_full_f"]:
                self.assertEqual(r.history[-1][key], summary[key])

    def test_initial_optimum_and_iteration_limit(self):
        p = ExponentialSum()
        optimum = gradient_descent(p, p.x_star, "exact")
        self.assertEqual(optimum.summary["iterations"], 0)
        self.assertEqual(optimum.summary["status"], "converged")
        self.assertEqual(optimum.summary["n_f"], 1)
        self.assertEqual(optimum.summary["n_g"], 1)
        limited = gradient_descent(p, [1, 1], "exact", max_updates=0)
        self.assertEqual(limited.summary["status"], "max_updates")
        self.assertEqual(limited.summary["iterations"], 0)

    def test_invalid_inputs(self):
        for gamma in [0, -1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                Quadratic(gamma)
        p = ExponentialSum()
        for alpha, beta in [(0, 0.5), (0.5, 0.5), (0.1, 0), (0.1, 1)]:
            with self.assertRaises(ValueError):
                armijo_backtracking(p, [1, 1], p.gradient([1, 1]), alpha=alpha, beta=beta)
        with self.assertRaises(ValueError):
            gradient_descent(p, [1, 1], "newton")
        with self.assertRaises(ValueError):
            gradient_descent(p, [1, 1], "exact", grad_tol=0)


if __name__ == "__main__":
    unittest.main()
