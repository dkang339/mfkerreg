"""MAROM"""

from dataclasses import dataclass
import numpy as np



@dataclass
class AlignedPOD:
    """Aligned high/low latent representations and reconstruction models."""

    high_coeff: np.ndarray
    low_coeff: np.ndarray
    high_model: "PODModel"
    low_model: "PODModel"
    n_modes: int
    high_energy: float
    low_energy: float
    alignment_error: float


@dataclass
class PODModel:
    """POD reduction and reconstruction."""

    basis: np.ndarray
    xoffset: np.ndarray
    retained_energy: float

    def compress(self, fields: np.ndarray) -> np.ndarray:
        return (fields - self.xoffset) @ self.basis.T

    def expand(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients @ self.basis + self.xoffset


def _pod(fields: np.ndarray, n_modes: int | None = None, energy: float | None = None):
    mean_field = fields.mean(axis=0)
    _, singular_values, basis = np.linalg.svd(fields - mean_field, full_matrices=False)
    cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    if n_modes is None:
        n_modes = int(np.searchsorted(cumulative_energy, energy, side="left") + 1)
    model = PODModel(basis[:n_modes], mean_field, float(cumulative_energy[n_modes - 1]))
    return model.compress(fields), model


def fit_aligned_pod(
    high_fields: np.ndarray, low_fields: np.ndarray, energy: float = 0.999999
) -> AlignedPOD:
    """Fit MAROM.

    Apply POD to high-fidelity data and choose the modes that satisfy the 
    energy criteria. These high-fidelity modes are used for reconstruction later.
    Apply POD to low-fidelity data, choose the same number of POD modes, and obtain 
    low-fidelity POD coefficients. Low-fidelity coefficients are then aligned into
    the high-fidelity latent coordinates using Procrustes analysis.
    """
    _, energy_model = _pod(high_fields, energy=energy)
    n_modes = energy_model.basis.shape[0]
    high_coeff, high_model = _pod(high_fields, n_modes=n_modes)
    low_coeff, low_model = _pod(low_fields, n_modes=n_modes)

    paired_low = low_coeff[: len(high_coeff)]
    high_mean = high_coeff.mean(axis=0)
    low_mean = paired_low.mean(axis=0)
    centered_high = high_coeff - high_mean
    centered_low = low_coeff - low_mean
    left, singular_values, right = np.linalg.svd(
        centered_low[: len(high_coeff)].T @ centered_high, full_matrices=False
    )
    rotation = left @ right
    scale = singular_values.sum() / np.linalg.norm(
        centered_low[: len(high_coeff)]
    ) ** 2
    low_coeff = scale * centered_low @ rotation
    alignment_error = float(
        np.linalg.norm(centered_high - low_coeff[: len(high_coeff)])
        / np.sqrt(len(high_coeff))
    )
    return AlignedPOD(
        high_coeff=high_coeff,
        low_coeff=low_coeff,
        high_model=high_model,
        low_model=low_model,
        n_modes=n_modes,
        high_energy=high_model.retained_energy,
        low_energy=low_model.retained_energy,
        alignment_error=alignment_error,
    )
