"""Visualization helpers for the CRM wing stress-field example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _display_mesh(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, rotation = np.linalg.svd(centered, full_matrices=False)
    return centered @ rotation.T


def _gapped_faces(points: np.ndarray, elements: np.ndarray, gap: float) -> np.ndarray:
    faces = points[elements].copy()
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    alignment = np.abs(normals[:, 2]) / np.maximum(
        np.linalg.norm(normals, axis=1), np.finfo(float).eps
    )
    skin = alignment > 0.65
    side_a = skin & (normals[:, 2] >= 0.0)
    side_b = skin & ~side_a
    if faces[side_a, :, 2].mean() > faces[side_b, :, 2].mean():
        upper_skin, lower_skin = side_a, side_b
    else:
        upper_skin, lower_skin = side_b, side_a
    separation = gap * np.ptp(points[:, 1])
    faces[upper_skin, :, 2] += separation
    faces[lower_skin, :, 2] -= separation
    return faces


def _add_surface(axis, faces, values, cmap, normalizer) -> None:
    axis.add_collection3d(
        Poly3DCollection(
            faces,
            array=values,
            cmap=cmap,
            norm=normalizer,
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
        )
    )


def plot_stress_fields(
    points: np.ndarray,
    elements: np.ndarray,
    truth: np.ndarray,
    sf_prediction: np.ndarray,
    mf_prediction: np.ndarray,
    output_path: Path,
    title: str,
    n_high: int,
    n_low: int,
    gap: float = 0.9,
    elevation: float = 30.0,
    azimuth: float = -110.0,
) -> None:
    """Save single- and multifidelity stress predictions beside their errors."""
    faces = _gapped_faces(_display_mesh(points), elements, gap)
    sf_error = np.abs(sf_prediction - truth)
    mf_error = np.abs(mf_prediction - truth)
    stress_norm = Normalize(
        vmin=min(truth.min(), sf_prediction.min(), mf_prediction.min()),
        vmax=max(truth.max(), sf_prediction.max(), mf_prediction.max()),
    )
    error_norm = Normalize(vmin=0.0, vmax=max(sf_error.max(), mf_error.max()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), subplot_kw={"projection": "3d"})
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.09, top=0.95, wspace=-0.40, hspace=-0.16)
    _add_surface(axes[0, 0], faces, sf_prediction, "viridis", stress_norm)
    _add_surface(axes[1, 0], faces, mf_prediction, "viridis", stress_norm)
    _add_surface(axes[0, 1], faces, sf_error, "magma", error_norm)
    _add_surface(axes[1, 1], faces, mf_error, "magma", error_norm)
    axes[0, 0].set_title(f"Single fidelity prediction (N_HF = {n_high})", y=0.84, pad=20)
    axes[1, 0].set_title(
        f"Multifidelity prediction (N_HF = {n_high}, N_LF = {n_low})", y=0.84, pad=20
    )
    axes[0, 1].set_title("Absolute error of single fidelity model", y=0.84, pad=20)
    axes[1, 1].set_title("Absolute error of multifidelity model", y=0.84, pad=20)
    visible_points = faces.reshape(-1, 3)
    spans = np.ptp(visible_points, axis=0)
    limits = np.column_stack((visible_points.min(axis=0), visible_points.max(axis=0)))
    for axis in axes.flat:
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_box_aspect(spans, zoom=1.3)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
    stress_cax = fig.add_axes([0.16, 0.035, 0.32, 0.025])
    error_cax = fig.add_axes([0.55, 0.035, 0.32, 0.025])
    fig.colorbar(
        ScalarMappable(norm=stress_norm, cmap="viridis"),
        cax=stress_cax,
        orientation="horizontal",
        label="von Mises stress (psi)",
    )
    fig.colorbar(
        ScalarMappable(norm=error_norm, cmap="magma"),
        cax=error_cax,
        orientation="horizontal",
        label="absolute error (psi)",
    )
    fig.suptitle(title, y=0.99, fontsize=18)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
