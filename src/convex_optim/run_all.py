"""执行两题的正式实验，并把真实迭代轨迹与汇总保存为 JSON / CSV。

主停止条件统一为 ||grad f|| <= gradient_tolerance。
第一题的 relative_gap_diagnostic_threshold 只用于统计首达轮，不改变运行。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .problems import Quadratic, ExponentialSum
from .optimize import gradient_descent

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_csv(path: Path, rows: list[dict]) -> None:
    """空值保留为空格，数字不进行用于展示的四舍五入。"""
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    def numpy_to_python(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                   default=numpy_to_python) + "\n", encoding="utf-8"
    )


def quadratic_theory(gamma: float, grad_tol: float, relative_tol: float) -> dict:
    """只用于验收的闭式计数；不会用它生成优化轨迹。"""
    r = abs((gamma - 1.0) / (gamma + 1.0))
    initial_grad = math.sqrt(2.0) * gamma
    if initial_grad <= grad_tol:
        kg = 0
    elif gamma == 1.0:
        kg = 1
    else:
        kg = math.ceil(math.log(grad_tol / initial_grad) / math.log(r))
    kr = 0 if relative_tol >= 1 else (
        1 if gamma == 1 else math.ceil(math.log(relative_tol) / (2 * math.log(r)))
    )
    return {"condition_number": max(gamma, 1.0 / gamma), "rho": r*r,
            "theoretical_gradient_iterations": kg,
            "theoretical_relative_hit_iterations": kr}


def first_hit(history: list[dict], field: str, threshold: float):
    return next((int(row["k"]) for row in history if row[field] <= threshold), None)


def analyze_run(run: dict, problem) -> dict:
    """优化运行结束后的独立数学检查，不计入算法评估次数。"""
    history = run["history"]
    points = np.array([[r["x1"], r["x2"]] for r in history])
    gaps = np.array([r["gap"] for r in history])
    diagnostics = {
        "run_id": run["run_id"],
        "updates_equal_saved_points_minus_one": len(history)-1 == run["summary"]["iterations"],
        "all_values_finite": bool(np.isfinite(points).all() and np.isfinite(gaps).all()),
        "max_gap_increase": max(0.0, float(np.max(np.diff(gaps)))) if len(gaps)>1 else 0.0,
    }
    if run["problem"] == "quadratic":
        gamma = run["gamma"]
        q = (gamma-1)/(gamma+1)
        expected = np.array([
            [gamma*q**k, (-q)**k] if k else [gamma, 1.0]
            for k in range(len(history))
        ])
        denominators = np.maximum(np.linalg.norm(expected, axis=1), 1e-300)
        relative_errors = np.linalg.norm(points-expected, axis=1)/denominators
        diagnostics["max_relative_closed_form_point_error"] = float(relative_errors.max())
        diagnostics["theory_gradient_count_matches"] = (
            run["summary"]["iterations"] == run["summary"]["theoretical_gradient_iterations"])
        diagnostics["theory_relative_count_matches"] = (
            run["summary"]["first_relative_hit_iterations"] ==
            run["summary"]["theoretical_relative_hit_iterations"])
    if run["method"] in {"quadratic_exact", "exact"}:
        cosines = []
        for left, right in zip(points[:-1], points[1:]):
            ga, gb = problem.gradient(left), problem.gradient(right)
            scale = float(np.linalg.norm(ga)*np.linalg.norm(gb))
            if scale > 0:
                cosines.append(abs(float(ga@gb))/scale)
        diagnostics["max_abs_adjacent_gradient_cosine"] = max(cosines, default=0.0)
    elif run["method"] == "backtracking":
        ratios = []
        powers = []
        for row, x, xp in zip(history[:-1], points[:-1], points[1:]):
            # 用独立 gap 恒等式求差，检验充分下降；不改变算法搜索。
            delta = problem.gap(xp)-problem.gap(x)
            denom = row["step"]*row["grad_norm"]**2
            ratios.append(-float(delta)/denom)
            powers.append(abs(math.log(row["step"])/math.log(run["beta"])-
                              round(math.log(row["step"])/math.log(run["beta"]))))
        diagnostics["min_accepted_decrease_ratio"] = min(ratios, default=None)
        diagnostics["min_armijo_ratio_slack"] = min(ratios, default=run["alpha"])-run["alpha"]
        diagnostics["max_distance_to_integer_beta_power"] = max(powers, default=0.0)
    return diagnostics


def run_experiments(config_path: Path, output_dir: Path, figures_dir: Path,
                    make_plots: bool = True) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "trajectories"
    traces_dir.mkdir(exist_ok=True)
    grad_tol = config["gradient_tolerance"]
    common = {"grad_tol": grad_tol, "max_updates": config["max_updates"]}
    runs, checks, threshold_rows = [], [], []

    def save_one(problem, x0, method, run_id, gamma=None, alpha=None, beta=None):
        kwargs = dict(common)
        if alpha is not None: kwargs["alpha"] = alpha
        if beta is not None: kwargs["beta"] = beta
        if method == "exact":
            kwargs["exact_xtol"] = config["exponential"]["exact_xtol"]
            kwargs["exact_rtol"] = config["exponential"]["exact_rtol"]
        result = gradient_descent(problem, x0, method, **kwargs)
        summary = dict(result.summary)
        summary.update(run_id=run_id, problem="quadratic" if gamma is not None else "exponential",
                       method=method, gamma=gamma, alpha=alpha, beta=beta)
        if gamma is not None:
            relative_tol = config["quadratic"]["relative_gap_diagnostic_threshold"]
            summary.update(quadratic_theory(gamma, grad_tol, relative_tol))
            summary["first_relative_hit_iterations"] = first_hit(result.history, "relative_gap", relative_tol)
        run = {"run_id": run_id, "problem": summary["problem"], "method": method,
               "gamma": gamma, "alpha": alpha, "beta": beta,
               "history": result.history, "summary": summary}
        runs.append(run)
        checks.append(analyze_run(run, problem))
        write_csv(traces_dir/f"{run_id}.csv", result.history)
        for threshold in config["gradient_threshold_diagnostics"]:
            threshold_rows.append({"run_id": run_id, "gradient_threshold": threshold,
                                   "first_hit_iterations": first_hit(result.history, "grad_norm", threshold)})
        print(f"{run_id:30s} K={summary['iterations']:4d}  "
              f"|g|={result.history[-1]['grad_norm']:.3e}  status={summary['status']}")

    for gamma in config["quadratic"]["gammas"]:
        gamma_name = "one_third" if math.isclose(gamma, 1/3, rel_tol=0, abs_tol=1e-15) else f"{gamma:g}".replace(".", "p")
        save_one(Quadratic(gamma), [gamma, 1.0], "quadratic_exact", f"quad_gamma_{gamma_name}", gamma=gamma)
    x0 = config["exponential"]["initial_point"]
    save_one(ExponentialSum(), x0, "exact", "exp_exact")
    for alpha in config["exponential"]["alphas"]:
        for beta in config["exponential"]["betas"]:
            run_id = f"exp_a{alpha:g}_b{beta:g}".replace(".", "p")
            save_one(ExponentialSum(), x0, "backtracking", run_id, alpha=alpha, beta=beta)

    for group in ["quadratic", "exponential"]:
        write_csv(output_dir/f"{group}_summary.csv", [r["summary"] for r in runs if r["problem"]==group])
    write_csv(output_dir/"gradient_threshold_diagnostics.csv", threshold_rows)
    # 两个 alpha 的相同 beta 轨迹，以及局部二次模型，均直接从实际日志检验。
    def select_exp(alpha, beta):
        return next(r for r in runs if r["problem"]=="exponential" and r["alpha"]==alpha and r["beta"]==beta)
    left, right = select_exp(.01,.3), select_exp(.1,.3)
    xy = lambda r: np.array([[h['x1'],h['x2']] for h in r['history']])
    same = xy(left).shape==xy(right).shape and np.array_equal(xy(left),xy(right))
    accepted_ratios = [h["normalized_decrease"] for h in left["history"][:-1]]
    rejected_ratios = [h["prev_rejected_normalized_decrease"] for h in left["history"][:-1]
                       if h["prev_rejected_normalized_decrease"] is not None]
    fstar = float(ExponentialSum().f_star)
    mechanism = {"same_trajectory_alpha_001_01_beta_03": bool(same),
                 "same_trajectory_accepted_ratio_min": min(accepted_ratios),
                 "same_trajectory_accepted_ratio_max": max(accepted_ratios),
                 "same_trajectory_last_rejected_ratio_min": min(rejected_ratios),
                 "same_trajectory_last_rejected_ratio_max": max(rejected_ratios),
                 "f_star": fstar, "lambda2_at_optimum": 4.5*fstar,
                 "local_axis2_stability_upper_step": 2/(4.5*fstar), "tail_checks": []}
    for alpha,beta in [(0.01,.7),(0.01,.9),(.1,.7),(.3,.7)]:
        run=select_exp(alpha,beta)
        history=run["history"]
        tail=history[-11:-1]
        steps=np.array([h["step"] for h in tail])
        actual=np.array([history[j+1]["gap"]/history[j]["gap"]
                         for j in range(max(0,len(history)-11),len(history)-1)])
        step=float(np.median(steps))
        constant_step=bool(np.allclose(steps,steps[0],rtol=1e-13,atol=0))
        mechanism["tail_checks"].append({"run_id":run["run_id"],"last_ten_steps_min":float(steps.min()),
            "last_ten_steps_max":float(steps.max()),"median_step":step,
            "last_ten_steps_constant":constant_step,
            "axis2_multiplier_if_fixed_step":1-step*4.5*fstar if constant_step else None,
            "predicted_axis2_gap_ratio_if_dominant":(1-step*4.5*fstar)**2 if constant_step else None,
            "last_ten_observed_gap_ratio_median":float(np.median(actual))})
    write_json(output_dir/"mechanism_checks.json", mechanism)
    write_json(output_dir/"validation.json", {"checks":checks})
    import matplotlib
    source_hashes = {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((PROJECT_ROOT/"src").rglob("*.py"))}
    payload={"schema_version":1,"config":config,"metadata":{
        "generated_utc":datetime.now(timezone.utc).isoformat(), "python":platform.python_version(),
        "platform":platform.platform(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,
        "source_sha256":source_hashes,
        "analysis_evaluations_excluded_from_optimizer_counts":True},"runs":runs}
    write_json(output_dir/"experiment_data.json", payload)
    write_json(output_dir/"run_metadata.json", payload["metadata"])
    write_json(output_dir/"used_config.json", config)
    if make_plots:
        from .plot_results import generate_figures
        generate_figures(output_dir/"experiment_data.json",figures_dir)
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/experiments.json")
    parser.add_argument("--output",type=Path,default=PROJECT_ROOT/"results")
    parser.add_argument("--figures",type=Path,default=PROJECT_ROOT/"figures")
    parser.add_argument("--no-plots",action="store_true")
    args=parser.parse_args()
    run_experiments(args.config,args.output,args.figures,not args.no_plots)


if __name__=="__main__":
    main()
