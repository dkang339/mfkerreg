"""Compare KR, MFKR, and KRR on the synthetic examples.

The benchmark intentionally includes bandwidth-selection time because that is
the main practical concern for the current MFKR implementation.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from kernels import ardmatern32  # noqa: E402
from kr import eval_kr, eval_mfkr, find_mfsigma, find_sigma  # noqa: E402
from mfmc import alloc  # noqa: E402


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    low: float
    high: float
    stats_path: Path

    def high_fidelity(self, x: np.ndarray) -> np.ndarray:
        if self.name == "exponential":
            return np.exp(x[:, 0])
        if self.name == "forrester":
            d = 2
            a, b, c = 20, 0.2, 2 * np.pi
            return (
                -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
                - np.exp(np.sum(np.cos(c * x), axis=1) / d)
                + a
                + np.exp(1)
            )
        raise ValueError(f"Unknown problem {self.name!r}.")

    def low_fidelity(self, x: np.ndarray) -> np.ndarray:
        if self.name == "exponential":
            return 0.9 * np.sqrt(np.exp(x[:, 0]))
        if self.name == "forrester":
            d = 2
            a, b, c = 20, 0.2, 2 * np.pi
            return (
                -a * np.exp(-0.9 * b * np.sqrt(np.sum(x**2, axis=1) / d))
                - np.exp(np.sum(np.sin(c * x), axis=1) / d)
                + a
                + np.exp(1)
                + 0.1 * x[:, 0]
            )
        raise ValueError(f"Unknown problem {self.name!r}.")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.ravel() - y_pred.ravel()) ** 2))


def krr_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sigma: float,
    ridge: float,
) -> np.ndarray:
    kernel_train = ardmatern32(x_train, x_train, sigma)
    kernel_test = ardmatern32(x_test, x_train, sigma)
    eye = np.eye(len(x_train))
    alpha = la.solve(kernel_train + ridge * eye, y_train, assume_a="pos")
    return kernel_test @ alpha


def krr_loocv_error(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma: float,
    ridge: float,
) -> float:
    kernel_train = ardmatern32(x_train, x_train, sigma)
    system = kernel_train + ridge * np.eye(len(x_train))
    factor = la.cho_factor(system, lower=True, check_finite=False)
    alpha = la.cho_solve(factor, y_train, check_finite=False)
    inverse = la.cho_solve(factor, np.eye(len(x_train)), check_finite=False)
    inverse_diag = np.maximum(np.diag(inverse), np.finfo(float).eps)
    loo_pred = y_train - alpha / inverse_diag[:, None]
    return mse(y_train, loo_pred)


def find_krr_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sigma_grid: np.ndarray,
    ridge_grid: np.ndarray,
) -> tuple[float, float]:
    best_error = np.inf
    best_sigma = float(sigma_grid[0])
    best_ridge = float(ridge_grid[0])
    for sigma in sigma_grid:
        for ridge in ridge_grid:
            err = krr_loocv_error(x_train, y_train, float(sigma), float(ridge))
            if err < best_error:
                best_error = err
                best_sigma = float(sigma)
                best_ridge = float(ridge)
    return best_sigma, best_ridge


def make_test_points(problem: Problem, n_test: int) -> np.ndarray:
    if problem.dim == 1:
        return np.linspace(problem.low, problem.high, n_test).reshape(-1, 1)

    side = int(np.sqrt(n_test))
    values = np.linspace(problem.low, problem.high, side)
    x1, x2 = np.meshgrid(values, values, indexing="ij")
    return np.stack([x1.ravel(), x2.ravel()], axis=1)


def run_one(
    problem: Problem,
    budget: int,
    reps: int,
    n_test: int,
    rng: np.random.Generator,
    sigma_grid: np.ndarray,
    ridge_grid: np.ndarray,
) -> dict[str, float]:
    stats = np.load(problem.stats_path)
    std = stats["sigma"]
    rho = stats["rho"]
    costs = np.array([1.0, 0.001])
    alpha = np.zeros(2)
    alpha[1], sample_counts, _ = alloc(std, rho, costs, budget)

    x_test = make_test_points(problem, n_test)
    y_test = problem.high_fidelity(x_test).reshape(-1, 1)

    sf_mse, sf_time = [], []
    mf_mse, mf_time = [], []
    krr_mse, krr_time = [], []

    for _ in range(reps):
        x_mf = rng.uniform(problem.low, problem.high, (sample_counts[-1], problem.dim))
        y_mf = np.zeros((sample_counts[-1], 1, 2))
        y_mf[: sample_counts[0], :, 0] = problem.high_fidelity(
            x_mf[: sample_counts[0]]
        ).reshape(-1, 1)
        y_mf[: sample_counts[1], :, 1] = problem.low_fidelity(
            x_mf[: sample_counts[1]]
        ).reshape(-1, 1)

        x_sf = rng.uniform(problem.low, problem.high, (budget, problem.dim))
        y_sf = problem.high_fidelity(x_sf).reshape(-1, 1)

        start = time.perf_counter()
        sigma_sf = find_sigma(x_sf, y_sf, kernel=ardmatern32, ard=True)
        y_pred_sf = eval_kr(x_test, x_sf, y_sf, sigma_sf, kernel=ardmatern32)
        sf_time.append(time.perf_counter() - start)
        sf_mse.append(mse(y_test, y_pred_sf))

        start = time.perf_counter()
        sigma_mf = find_mfsigma(x_mf, y_mf, sample_counts, kernel=ardmatern32, ard=True)
        y_pred_mf = eval_mfkr(
            x_test, x_mf, y_mf, sigma_mf, sample_counts, alpha, kernel=ardmatern32
        )
        mf_time.append(time.perf_counter() - start)
        mf_mse.append(mse(y_test, y_pred_mf))

        start = time.perf_counter()
        sigma_krr, ridge = find_krr_params(x_sf, y_sf, sigma_grid, ridge_grid)
        y_pred_krr = krr_predict(x_sf, y_sf, x_test, sigma_krr, ridge)
        krr_time.append(time.perf_counter() - start)
        krr_mse.append(mse(y_test, y_pred_krr))

    return {
        "budget": float(budget),
        "mf_high": float(sample_counts[0]),
        "mf_low": float(sample_counts[1]),
        "sf_mse": float(np.mean(sf_mse)),
        "sf_time": float(np.mean(sf_time)),
        "mf_mse": float(np.mean(mf_mse)),
        "mf_time": float(np.mean(mf_time)),
        "krr_mse": float(np.mean(krr_mse)),
        "krr_time": float(np.mean(krr_time)),
    }


def print_table(problem_name: str, rows: list[dict[str, float]]) -> None:
    print(f"\n{problem_name}")
    print(
        "budget  MF samples   SF-KR MSE/time     MFKR MSE/time      "
        "KRR MSE/time"
    )
    for row in rows:
        print(
            f"{int(row['budget']):>6}  "
            f"{int(row['mf_high']):>4}/{int(row['mf_low']):<5}  "
            f"{row['sf_mse']:.3e}/{row['sf_time']:.4f}s  "
            f"{row['mf_mse']:.3e}/{row['mf_time']:.4f}s  "
            f"{row['krr_mse']:.3e}/{row['krr_time']:.4f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["exponential", "forrester"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-test", type=int, default=225)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    problems = {
        "exponential": Problem(
            "exponential", 1, 0.0, 5.0, ROOT_DIR / "data" / "exponential" / "stats_exp.npz"
        ),
        "forrester": Problem(
            "forrester", 2, -32.768, 32.768, ROOT_DIR / "data" / "forrest" / "stats_forrest.npz"
        ),
    }
    sigma_grid = np.logspace(-2, 2, 9)
    ridge_grid = np.logspace(-10, -2, 9)
    rng = np.random.default_rng(args.seed)

    for problem_name in args.problems:
        if problem_name not in problems:
            raise ValueError(f"Unknown problem {problem_name!r}.")
        rows = [
            run_one(
                problems[problem_name],
                budget,
                args.reps,
                args.n_test,
                rng,
                sigma_grid,
                ridge_grid,
            )
            for budget in args.budgets
        ]
        print_table(problem_name, rows)


if __name__ == "__main__":
    main()
