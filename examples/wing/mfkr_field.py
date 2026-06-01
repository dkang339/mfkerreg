"""Predict wing stress fields using MAROM and MF kernel regression.

Following the MAROM workflow, independent high- and
low-fidelity POD spaces are fitted and paired POD coefficients are aligned
with scaled orthogonal Procrustes. MF kernel regression replaces the
coefficient-surrogate stage. Reconstruction uses the high-fidelity POD basis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "wing"
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from kr import eval_kr, eval_mfkr, find_mfsigma  # noqa: E402
from marom_alignment import fit_aligned_pod  # noqa: E402
from wing_plot import plot_stress_fields  # noqa: E402


def read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read random unique HDF5 rows and preserve the requested order."""
    order = np.argsort(indices)
    sorted_rows = dataset[indices[order]]
    return sorted_rows[np.argsort(order)]


def coefficient_alpha(high_coeff: np.ndarray, low_coeff: np.ndarray) -> np.ndarray:
    """Compute a control-variate coefficient independently for each POD mode."""
    high_delta = high_coeff - high_coeff.mean(axis=0)
    low_delta = low_coeff - low_coeff.mean(axis=0)
    variance = np.sum(low_delta**2, axis=0)
    covariance = np.sum(high_delta * low_delta, axis=0)
    return np.divide(covariance, variance, out=np.zeros_like(covariance), where=variance > 0)


def relative_l2(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(truth, axis=1)
    return np.linalg.norm(prediction - truth, axis=1) / np.maximum(
        denominator, np.finfo(float).eps
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--low-fidelity", choices=("rib", "grid"), default="grid")
    parser.add_argument("--n-high", type=int, default=100, help="High-fidelity training count.")
    parser.add_argument("--n-low", type=int, default=300, help="Low-fidelity training count.")
    parser.add_argument("--n-test", type=int, default=100, help="Held-out high-fidelity count.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--energy", type=float, default=0.999999, help="POD energy fraction.")
    parser.add_argument(
        "--plot-test-index",
        type=int,
        default=-1,
        help="Index within testing set to plot; -1 selects worst relative MF error.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.9,
        help="Upper/lower skin separation as a fraction of displayed wing chord.",
    )
    parser.add_argument("--view-elev", type=float, default=30.0, help="3D plot elevation angle.")
    parser.add_argument("--view-azim", type=float, default=-110.0, help="3D plot azimuth angle.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    if not 0.0 < args.energy <= 1.0:
        raise ValueError("--energy must be in (0, 1].")
    if args.gap < 0.0:
        raise ValueError("--gap cannot be negative.")
    if args.n_high > args.n_low:
        raise ValueError("--n-low must be at least --n-high for nested MF samples.")

    high_path = args.data_dir / "highfi_field.h5"
    low_path = args.data_dir / f"lowfi_{args.low_fidelity}_field.h5"
    if not high_path.exists() or not low_path.exists():
        raise FileNotFoundError("Run examples/wing/preproc_wing.py before field prediction.")

    with h5py.File(high_path, "r") as high, h5py.File(low_path, "r") as low:
        n_samples = high["input"].shape[0]
        if args.n_test + args.n_low > n_samples:
            raise ValueError("--n-test + --n-low cannot exceed the available samples.")
        if not np.allclose(high["input"][:], low["input"][:]):
            raise ValueError("High- and low-fidelity design inputs are not paired.")

        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(n_samples)
        test_indices = permutation[: args.n_test]
        training_pool = permutation[args.n_test : args.n_test + args.n_low]
        high_indices = training_pool[: args.n_high]
        low_indices = training_pool
        high_fields = read_rows(high["output"], high_indices)
        test_fields = read_rows(high["output"], test_indices)
        low_fields = read_rows(low["output"], low_indices)
        x_train = read_rows(high["input"], low_indices)
        x_test = read_rows(high["input"], test_indices)
        nodes = high["node"][:]
        elements = high["elem"][:]

    reduction = fit_aligned_pod(high_fields, low_fields, args.energy)
    high_coeff = reduction.high_coeff
    aligned_low_coeff = reduction.low_coeff

    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    x_train_scaled = (x_train - x_mean) / x_scale
    x_test_scaled = (x_test - x_mean) / x_scale
    sample_counts = np.array([args.n_high, args.n_low])
    outputs = np.zeros((args.n_low, reduction.n_modes, 2))
    outputs[: args.n_high, :, 0] = high_coeff
    outputs[:, :, 1] = aligned_low_coeff
    alpha = np.zeros((2, reduction.n_modes))
    alpha[1] = coefficient_alpha(high_coeff, aligned_low_coeff[: args.n_high])

    sigma = find_mfsigma(x_train_scaled, outputs, sample_counts)
    predicted_coeff = eval_mfkr(
        x_test_scaled, x_train_scaled, outputs, sigma, sample_counts, alpha
    )
    sf_coeff = eval_kr(
        x_test_scaled, x_train_scaled[: args.n_high], high_coeff, sigma.sigma0
    )
    prediction = reduction.high_model.expand(predicted_coeff)
    sf_prediction = reduction.high_model.expand(sf_coeff)
    mf_relative_error = relative_l2(test_fields, prediction)
    sf_relative_error = relative_l2(test_fields, sf_prediction)
    projection_error = relative_l2(
        test_fields, reduction.high_model.expand(reduction.high_model.compress(test_fields))
    )

    plot_index = (
        int(np.argmax(mf_relative_error)) if args.plot_test_index < 0 else args.plot_test_index
    )
    if not 0 <= plot_index < args.n_test:
        raise ValueError("--plot-test-index must refer to an item in the test set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_label = f"lowfi_{args.low_fidelity}"
    plot_path = args.output_dir / f"mfkr_field_{low_label}.png"
    result_path = args.output_dir / f"mfkr_field_{low_label}.npz"
    plot_stress_fields(
        nodes,
        elements,
        test_fields[plot_index],
        sf_prediction[plot_index],
        prediction[plot_index],
        plot_path,
        "Comparison of von Mises stress distributions",
        args.n_high,
        args.n_low,
        args.gap,
        args.view_elev,
        args.view_azim,
    )
    np.savez_compressed(
        result_path,
        test_indices=test_indices,
        high_indices=high_indices,
        low_indices=low_indices,
        high_basis=reduction.high_model.basis,
        high_mean=reduction.high_model.xoffset,
        low_basis=reduction.low_model.basis,
        low_mean=reduction.low_model.xoffset,
        predicted_coeff=predicted_coeff,
        mf_relative_error=mf_relative_error,
        sf_relative_error=sf_relative_error,
        projection_relative_error=projection_error,
        alpha=alpha[1],
        high_retained_energy=reduction.high_energy,
        low_retained_energy=reduction.low_energy,
        alignment_error=reduction.alignment_error,
    )

    print(
        f"POD modes retained: {reduction.n_modes} "
        f"(HF energy {reduction.high_energy:.8f}, "
        f"LF energy {reduction.low_energy:.8f})"
    )
    print(f"Procrustes alignment RMSE: {reduction.alignment_error:.4e}")
    print(
        "MF relative L2 error: "
        f"mean={mf_relative_error.mean():.4e}, max={mf_relative_error.max():.4e}"
    )
    print(
        "SF relative L2 error: "
        f"mean={sf_relative_error.mean():.4e}, max={sf_relative_error.max():.4e}"
    )
    print(
        "POD projection relative L2 error: "
        f"mean={projection_error.mean():.4e}, max={projection_error.max():.4e}"
    )
    print(f"Saved {plot_path}")
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
