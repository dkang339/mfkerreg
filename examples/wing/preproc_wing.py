"""Preprocess the CRM wing datasets with the stress fields.

The source files contain disconnected structural components. This script
extracts the largest connected triangular component (the primary wing) and
writes field datasets needed for field prediction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "wing"
DATASETS = {
    "highfi": "crm_baseline_4DV_N1000_slim.h5",
    "lowfi_grid": "crm_coarse-grid_4DV_N1000_slim.h5",
    "lowfi_rib": "crm_coarse-ribs_4DV_N1000_slim.h5",
}


def primary_wing_mesh(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return primary-wing nodes, remapped triangles, and source-cell indices."""
    with h5py.File(raw_path, "r") as source:
        points = source["GRID/Points"][0]
        elements = source["GRID/Cells"][:].reshape(-1, 4)[:, 1:4]

    edge_to_elements: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for element_index, element in enumerate(elements):
        n0, n1, n2 = np.sort(element)
        for edge in ((n0, n1), (n0, n2), (n1, n2)):
            edge_to_elements[edge].append(element_index)

    rows: list[int] = []
    cols: list[int] = []
    for touching_elements in edge_to_elements.values():
        for i, left in enumerate(touching_elements):
            for right in touching_elements[i + 1 :]:
                rows.extend((left, right))
                cols.extend((right, left))

    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(elements), len(elements)),
    )
    _, labels = connected_components(graph)
    primary_label = np.bincount(labels).argmax()
    source_indices = np.flatnonzero(labels == primary_label)
    primary_elements = elements[source_indices]
    source_nodes = np.unique(primary_elements)
    remapped_elements = np.searchsorted(source_nodes, primary_elements)
    return points[source_nodes], remapped_elements, source_indices


def preprocess_dataset(raw_path: Path, output_path: Path, overwrite: bool = False) -> None:
    """Extract primary-wing stress fields and write one processed HDF5 file."""
    if output_path.exists() and not overwrite:
        print(f"Keeping existing file: {output_path}")
        return

    nodes, elements, source_indices = primary_wing_mesh(raw_path)
    with h5py.File(raw_path, "r") as source, h5py.File(output_path, "w") as target:
        stress = source["GRID/CELL_DATA/vonMisesStress_2p5g_psi"]
        inputs = source["STATE/DV_Values"][:]
        n_samples = stress.shape[0]
        n_cells = len(source_indices)
        target.attrs["source_file"] = raw_path.name
        target.attrs["field"] = "vonMisesStress_2p5g_psi"
        target.attrs["component"] = "largest connected primary wing"
        target.create_dataset("input", data=inputs, compression="gzip", compression_opts=4)
        target.create_dataset("node", data=nodes, compression="gzip", compression_opts=4)
        target.create_dataset("elem", data=elements, compression="gzip", compression_opts=4)
        output = target.create_dataset(
            "output",
            shape=(n_samples, n_cells),
            dtype=stress.dtype,
            chunks=(min(16, n_samples), min(4096, n_cells)),
            compression="gzip",
            compression_opts=4,
        )
        for start in range(0, n_samples, 16):
            stop = min(start + 16, n_samples)
            output[start:stop] = stress[start:stop, source_indices]

    print(
        f"Wrote {output_path} with {n_samples} fields on "
        f"{n_cells} primary-wing elements."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Raw/output data directory."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS) + ("all",),
        default="all",
        help="Dataset to preprocess.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace processed files.")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    names = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for output_stem, raw_filename in names.items():
        raw_path = args.data_dir / raw_filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw wing dataset not found: {raw_path}")
        preprocess_dataset(raw_path, args.data_dir / f"{output_stem}_field.h5", args.overwrite)


if __name__ == "__main__":
    main()
