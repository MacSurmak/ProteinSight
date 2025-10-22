# prepare_feature_cache.py
# -*- coding: utf-8 -*-
"""
Generates a complete, high-performance feature cache for protein structures.

This script employs an asynchronous producer-consumer architecture with
multiprocessing to efficiently convert protein-ligand structures into
multi-channel 3D feature tensors suitable for deep learning models.

The pipeline includes:
1.  Parsing PQR (protein) and PDB (ligand) files.
2.  Calculating 8 distinct physicochemical and geometric feature fields on the GPU
    using an optimized "splatting" technique with Numba and CuPy.
3.  Generating a target mask based on the ligand's position.
4.  Saving the resulting tensors to HDF5 files for fast access during training.
5.  Creating informative 2D visualizations of the feature fields.
"""

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

# Suppress Numba performance warnings for cleaner output
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_laplace as gaussian_laplace_gpu
    from numba import cuda
except ImportError:
    sys.exit(
        "Error: Required GPU libraries not found. "
        "Please install: pip install numba cupy-cuda12x pandas matplotlib scipy h5py rich"
    )

# --- Global Configuration ---
console = Console()

OUTPUT_ROOT = Path("feature_cache")
FORCE_RECACHE = False
NUM_PROCESS_WORKERS = 8

DATASET_CONFIGS = {
    "positive": {
        "pdb_dir": Path("positive_pdb_files"),
        "pqr_dir": Path("positive_pqr_files"),
        "csv_file": Path("protein_clustering_results/clustered_protein_report.csv"),
        "cache_dir": OUTPUT_ROOT / "positive_cache",
        "vis_dir": OUTPUT_ROOT / "positive_visualization",
    },
    "negative": {
        "pdb_dir": Path("negative_pdb_files"),
        "pqr_dir": Path("negative_pqr_files"),
        "csv_file": Path("negative_controls_results/negative_controls_results_report.csv"),
        "cache_dir": OUTPUT_ROOT / "negative_cache",
        "vis_dir": OUTPUT_ROOT / "negative_visualization",
    },
    "albumins": {
        "pdb_dir": Path("albumins_pdb_files"),
        "pqr_dir": Path("albumins_pqr_files"),
        "csv_file": Path("albumins_results/albumins_results_report.csv"),
        "cache_dir": OUTPUT_ROOT / "albumins_cache",
        "vis_dir": OUTPUT_ROOT / "albumins_visualization",
    },
}

# --- Physicochemical Constants ---
SASA_PROBE_RADIUS_A = 1.4
VOXEL_RESOLUTION = 0.5
CUDA_BLOCK_SIZE_1D = 256
CUDA_BLOCK_SIZE_3D = (8, 8, 8)
CHANNELS = [
    "occupancy", "hydrophobicity", "electrostatic", "aromatic_field",
    "shape_index", "hbond_donor", "hbond_acceptor", "sasa",
]
CHANNEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9,
    "TYR": -1.3, "VAL": 4.2,
}
AROMATIC_RES_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
}
HBOND_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "TRP": {"NE1"}, "ASN": {"ND2"},
    "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"},
    "TYR": {"OH"}, "CYS": {"SG"},
}
HBOND_ACCEPTORS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"}, "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
}
VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "DEFAULT": 1.70}
(ELEC_NORM, SHAPE_NORM, OCC_NORM, HB_D_NORM, HB_A_NORM, AROMA_NORM) = \
    (0.5, 0.5, 3.5, 0.5, 0.5, 0.5)


# --- Data Structures ---
@dataclass(frozen=True)
class Atom:
    """Represents a single atom with its properties."""
    coords: np.ndarray
    radius: float
    charge: float = 0.0
    hydrophobicity: float = 0.0
    is_aromatic: bool = False
    is_hbond_donor: bool = False
    is_hbond_acceptor: bool = False

@dataclass(frozen=True)
class GridParameters:
    """Defines the properties of the 3D grid."""
    dims: Tuple[int, int, int]
    origin: np.ndarray
    resolution: float


# --- CUDA Kernels ---
@cuda.jit
def splatting_kernel(
    out, grid_min, grid_dims, res, coords, charges, hydros,
    donor_flags, acceptor_flags, aromatic_flags, atom_radii
):
    """Projects atomic properties onto the grid using Gaussian splatting."""
    i = cuda.grid(1)
    if i >= coords.shape[0]:
        return

    radius = atom_radii[i]
    sigma_occ = radius * 0.9
    sigma_sq_occ = sigma_occ * sigma_occ
    max_cutoff = 12.0

    pos_x, pos_y, pos_z = coords[i, 0], coords[i, 1], coords[i, 2]
    min_x = max(0, int(((pos_x - max_cutoff) - grid_min[0]) / res))
    max_x = min(grid_dims[0], int(((pos_x + max_cutoff) - grid_min[0]) / res) + 1)
    min_y = max(0, int(((pos_y - max_cutoff) - grid_min[1]) / res))
    max_y = min(grid_dims[1], int(((pos_y + max_cutoff) - grid_min[1]) / res) + 1)
    min_z = max(0, int(((pos_z - max_cutoff) - grid_min[2]) / res))
    max_z = min(grid_dims[2], int(((pos_z + max_cutoff) - grid_min[2]) / res) + 1)

    for z_idx in range(min_z, max_z):
        for y_idx in range(min_y, max_y):
            for x_idx in range(min_x, max_x):
                v_pos_x = grid_min[0] + (x_idx + 0.5) * res
                v_pos_y = grid_min[1] + (y_idx + 0.5) * res
                v_pos_z = grid_min[2] + (z_idx + 0.5) * res
                dist_sq = ((v_pos_x - pos_x)**2 + (v_pos_y - pos_y)**2 + (v_pos_z - pos_z)**2)

                if dist_sq < (4 * sigma_occ)**2: # cutoff at 4*sigma
                    cuda.atomic.add(out, (0, z_idx, y_idx, x_idx), math.exp(-dist_sq / (2.0 * sigma_sq_occ)))
                if dist_sq < 64.0:
                    cuda.atomic.add(out, (1, z_idx, y_idx, x_idx), hydros[i] * math.exp(-dist_sq / 8.0))
                if dist_sq < 144.0:
                    cuda.atomic.add(out, (2, z_idx, y_idx, x_idx), charges[i] * math.exp(-dist_sq / 24.5))
                if aromatic_flags[i] and dist_sq < 25.0:
                    cuda.atomic.add(out, (3, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88))
                if donor_flags[i] and dist_sq < 25.0:
                    cuda.atomic.add(out, (5, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88))
                if acceptor_flags[i] and dist_sq < 25.0:
                    cuda.atomic.add(out, (6, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88))


@cuda.jit
def splatting_vdw_mask_kernel(mask_out, grid_min, grid_dims, res, atom_coords, atom_radii):
    """Creates a binary mask of the protein based on Van der Waals radii."""
    i = cuda.grid(1)
    if i >= atom_coords.shape[0]:
        return
    radius, radius_sq = atom_radii[i], atom_radii[i]**2
    pos_x, pos_y, pos_z = atom_coords[i, 0], atom_coords[i, 1], atom_coords[i, 2]
    min_x = max(0, int(((pos_x - radius) - grid_min[0]) / res))
    max_x = min(grid_dims[0], int(((pos_x + radius) - grid_min[0]) / res) + 1)
    min_y = max(0, int(((pos_y - radius) - grid_min[1]) / res))
    max_y = min(grid_dims[1], int(((pos_y + radius) - grid_min[1]) / res) + 1)
    min_z = max(0, int(((pos_z - radius) - grid_min[2]) / res))
    max_z = min(grid_dims[2], int(((pos_z + radius) - grid_min[2]) / res) + 1)
    for z_idx in range(min_z, max_z):
        for y_idx in range(min_y, max_y):
            for x_idx in range(min_x, max_x):
                v_pos_x = grid_min[0] + (x_idx + 0.5) * res
                v_pos_y = grid_min[1] + (y_idx + 0.5) * res
                v_pos_z = grid_min[2] + (z_idx + 0.5) * res
                if ((v_pos_x - pos_x)**2 + (v_pos_y - pos_y)**2 + (v_pos_z - pos_z)**2) <= radius_sq:
                    cuda.atomic.max(mask_out, (z_idx, y_idx, x_idx), 1.0)


@cuda.jit
def splatting_target_mask_kernel(
    target_mask_out, protein_mask_in, grid_min, grid_dims, res, ligand_coords, sigma_sq
):
    """Generates the target mask based on ligand atom positions."""
    i = cuda.grid(1)
    if i >= ligand_coords.shape[0]:
        return
    max_cutoff_target = 6.0
    pos_x, pos_y, pos_z = ligand_coords[i, 0], ligand_coords[i, 1], ligand_coords[i, 2]
    min_x = max(0, int(((pos_x - max_cutoff_target) - grid_min[0]) / res))
    max_x = min(grid_dims[0], int(((pos_x + max_cutoff_target) - grid_min[0]) / res) + 1)
    min_y = max(0, int(((pos_y - max_cutoff_target) - grid_min[1]) / res))
    max_y = min(grid_dims[1], int(((pos_y + max_cutoff_target) - grid_min[1]) / res) + 1)
    min_z = max(0, int(((pos_z - max_cutoff_target) - grid_min[2]) / res))
    max_z = min(grid_dims[2], int(((pos_z + max_cutoff_target) - grid_min[2]) / res) + 1)
    for z_idx in range(min_z, max_z):
        for y_idx in range(min_y, max_y):
            for x_idx in range(min_x, max_x):
                if protein_mask_in[z_idx, y_idx, x_idx] == 0:
                    continue
                v_pos_x = grid_min[0] + (x_idx + 0.5) * res
                v_pos_y = grid_min[1] + (y_idx + 0.5) * res
                v_pos_z = grid_min[2] + (z_idx + 0.5) * res
                dist_sq = ((v_pos_x - pos_x)**2 + (v_pos_y - pos_y)**2 + (v_pos_z - pos_z)**2)
                cuda.atomic.max(target_mask_out, (z_idx, y_idx, x_idx), math.exp(-dist_sq / sigma_sq))


@cuda.jit
def calculate_sasa_kernel(sasa_out, protein_mask_in):
    """Calculates a proxy for solvent-accessible surface area."""
    z, y, x = cuda.grid(3)
    dims = protein_mask_in.shape
    if z >= dims[0] or y >= dims[1] or x >= dims[2] or protein_mask_in[z, y, x] == 0:
        return
    accessible_neighbors = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if not (0 <= nz < dims[0] and 0 <= ny < dims[1] and 0 <= nx < dims[2]) or \
                   protein_mask_in[nz, ny, nx] == 0:
                    accessible_neighbors += 1
    sasa_out[z, y, x] = accessible_neighbors


@cuda.jit
def splatting_dilation_kernel(dilated_mask_out, protein_mask_in, probe_radius_voxels):
    """Performs a dilation of the protein mask by a given radius."""
    z, y, x = cuda.grid(3)
    dims = protein_mask_in.shape
    if z >= dims[0] or y >= dims[1] or x >= dims[2] or protein_mask_in[z, y, x] == 0:
        return
    r_int = int(probe_radius_voxels)
    r_sq = probe_radius_voxels * probe_radius_voxels
    for dz in range(-r_int, r_int + 1):
        for dy in range(-r_int, r_int + 1):
            for dx in range(-r_int, r_int + 1):
                if (dx*dx + dy*dy + dz*dz) > r_sq:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < dims[0] and 0 <= ny < dims[1] and 0 <= nx < dims[2]:
                    cuda.atomic.max(dilated_mask_out, (nz, ny, nx), 1.0)


class SplattingFieldCalculator:
    """Orchestrates the calculation of all feature fields for a structure."""

    def __init__(self, resolution: float):
        self.resolution = resolution

    @staticmethod
    def parse_protein_from_pqr(pqr_filepath: Path) -> List[Atom]:
        atoms = []
        with pqr_filepath.open("r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        res, name = line[17:20].strip(), line[12:16].strip()
                        coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        charge, radius = float(line[54:62]), float(line[62:68])
                        atoms.append(Atom(coords, radius, charge,
                            KYTE_DOOLITTLE.get(res, 0.0),
                            name in AROMATIC_RES_ATOMS.get(res, set()),
                            name in HBOND_DONORS.get(res, set()),
                            name in HBOND_ACCEPTORS.get(res, set())))
                    except (ValueError, IndexError):
                        continue
        return atoms

    @staticmethod
    def parse_ligand_from_pdb(pdb_filepath: Path, names: List[str]) -> List[Atom]:
        atoms = []
        with pdb_filepath.open("r") as f:
            for line in f:
                if line.startswith("HETATM") and line[17:20].strip() in names:
                    try:
                        coords = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        elem = (line[76:78].strip().upper() or line[12:14].strip().upper())
                        atoms.append(Atom(coords, VDW_RADII.get(elem, VDW_RADII["DEFAULT"])))
                    except (ValueError, IndexError):
                        continue
        return atoms

    def _get_grid_parameters(self, all_atoms: List[Atom]) -> GridParameters:
        if not all_atoms:
            raise ValueError("Atom list cannot be empty.")
        coords = np.array([a.coords for a in all_atoms])
        min_c, max_c = coords.min(axis=0) - 12.0, coords.max(axis=0) + 12.0
        dims = tuple(np.ceil((max_c - min_c) / self.resolution).astype(int))
        return GridParameters(dims, min_c, self.resolution)

    def calculate_all_fields(
        self, protein_atoms: List[Atom], ligand_atoms: List[Atom]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, GridParameters]]:
        if not protein_atoms:
            return None
        
        num_atoms = len(protein_atoms)
        grid_params = self._get_grid_parameters(protein_atoms + ligand_atoms)
        dims_zyx = (grid_params.dims[2], grid_params.dims[1], grid_params.dims[0])
        grid_min_d = cp.asarray(grid_params.origin, dtype=cp.float32)
        grid_dims_d = cp.asarray(grid_params.dims, dtype=cp.int32)
        
        # generate protein mask
        protein_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
        prot_coords_d = cp.asarray([a.coords for a in protein_atoms], dtype=cp.float32)
        prot_radii_d = cp.asarray([a.radius for a in protein_atoms], dtype=cp.float32)
        grid_size_1d = (num_atoms + CUDA_BLOCK_SIZE_1D - 1) // CUDA_BLOCK_SIZE_1D
        splatting_vdw_mask_kernel[grid_size_1d, CUDA_BLOCK_SIZE_1D](
            protein_mask_d, grid_min_d, grid_dims_d, self.resolution, prot_coords_d, prot_radii_d)
        
        # calculate primary features
        features_d = cp.zeros((len(CHANNELS), *dims_zyx), dtype=cp.float32)
        charges_d = cp.asarray([a.charge for a in protein_atoms], dtype=cp.float32)
        hydros_d = cp.asarray([a.hydrophobicity for a in protein_atoms], dtype=cp.float32)
        donor_d = cp.asarray([a.is_hbond_donor for a in protein_atoms], dtype=cp.bool_)
        acceptor_d = cp.asarray([a.is_hbond_acceptor for a in protein_atoms], dtype=cp.bool_)
        aromatic_d = cp.asarray([a.is_aromatic for a in protein_atoms], dtype=cp.bool_)
        splatting_kernel[grid_size_1d, CUDA_BLOCK_SIZE_1D](
            features_d, grid_min_d, grid_dims_d, self.resolution, prot_coords_d,
            charges_d, hydros_d, donor_d, acceptor_d, aromatic_d, prot_radii_d)
        
        # calculate derived features (shape index and sasa)
        features_d[CHANNEL_MAP["shape_index"]] = -gaussian_laplace_gpu(features_d[0].copy(), sigma=2.0)
        
        probe_radius_voxels = SASA_PROBE_RADIUS_A / self.resolution
        expanded_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
        grid_dim_3d = tuple((s + b - 1) // b for s, b in zip(dims_zyx, CUDA_BLOCK_SIZE_3D))
        splatting_dilation_kernel[grid_dim_3d, CUDA_BLOCK_SIZE_3D](expanded_mask_d, protein_mask_d, probe_radius_voxels)
        sasa_d = cp.zeros(dims_zyx, dtype=cp.float32)
        calculate_sasa_kernel[grid_dim_3d, CUDA_BLOCK_SIZE_3D](sasa_d, expanded_mask_d)
        features_d[CHANNEL_MAP["sasa"]] = sasa_d / 26.0
        
        # calculate target mask
        target_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
        if ligand_atoms:
            ligand_coords_d = cp.asarray([a.coords for a in ligand_atoms], dtype=cp.float32)
            ligand_grid_size_1d = (len(ligand_atoms) + CUDA_BLOCK_SIZE_1D - 1) // CUDA_BLOCK_SIZE_1D
            splatting_target_mask_kernel[ligand_grid_size_1d, CUDA_BLOCK_SIZE_1D](
                target_mask_d, protein_mask_d, grid_min_d, grid_dims_d,
                self.resolution, ligand_coords_d, 4.5)
        
        # normalize and finalize tensors
        features_d[0] = cp.tanh(features_d[0] / OCC_NORM)
        features_d[1] = (2 * (features_d[1] - (-4.5)) / 9.0) - 1
        features_d[2] = cp.tanh(features_d[2] / ELEC_NORM)
        features_d[3] = cp.tanh(features_d[3] / AROMA_NORM)
        features_d[4] = cp.tanh(features_d[4] / SHAPE_NORM)
        features_d[5] = cp.tanh(features_d[5] / HB_D_NORM)
        features_d[6] = cp.tanh(features_d[6] / HB_A_NORM)
        final_features = cp.nan_to_num(features_d).astype(cp.float16).get()
        final_mask = target_mask_d.get()
        
        cp.get_default_memory_pool().free_all_blocks()
        return final_features, final_mask, num_atoms, grid_params


def process_and_visualize_task(
    pdb_id, features, target_mask, cache_file, vis_dir, ligand_atoms, grid_params
):
    """Saves cache file and generates a visualization of feature slices."""
    with h5py.File(cache_file, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("target_mask", data=target_mask, compression="gzip")

    features_f32 = features.astype(np.float32)
    slice_title_suffix = ""
    # determine slice center: ligand's center of mass or grid center
    if ligand_atoms:
        center_coords = np.array([atom.coords for atom in ligand_atoms]).mean(axis=0)
        voxel_center = np.round((center_coords - grid_params.origin) / grid_params.resolution).astype(int)
        cx, cy, cz = voxel_center
        cz = np.clip(cz, 0, features.shape[1] - 1)
        cy = np.clip(cy, 0, features.shape[2] - 1)
        cx = np.clip(cx, 0, features.shape[3] - 1)
        slice_title_suffix = " (slice at ligand center)"
    else:
        cz, cy, cx = (np.array(features_f32.shape[1:]) // 2)
        slice_title_suffix = " (slice at grid center)"

    num_cols = len(CHANNELS) + 1
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 14), facecolor="black", constrained_layout=True)
    fig.suptitle(f"Orthogonal Field Slices for {pdb_id}{slice_title_suffix}", color="white", fontsize=22)
    
    slice_info = [("YZ", cz), ("XZ", cy), ("XY", cx)]
    col_titles = [ch.replace("_", " ").title() for ch in CHANNELS] + ["Target Mask"]
    cmaps = {"hydrophobicity": "seismic", "electrostatic": "bwr", "aromatic_field": "viridis",
             "shape_index": "PRGn", "hbond_donor": "Reds", "hbond_acceptor": "Blues",
             "occupancy": "plasma", "sasa": "magma"}
    
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, color="white", fontsize=16)
        
    for i, (plane_name, coord) in enumerate(slice_info):
        axes[i, 0].set_ylabel(f"{plane_name} (slice at {coord})", color="white", fontsize=14)
        for j, name in enumerate(CHANNELS):
            ax = axes[i, j]
            if plane_name == "YZ":   data = features_f32[j, coord, :, :].T
            elif plane_name == "XZ": data = features_f32[j, :, coord, :].T
            else:                    data = features_f32[j, :, :, coord].T
            vmin, vmax = (-1, 1) if name in ["hydrophobicity", "electrostatic", "shape_index"] else (0, 1)
            im = ax.imshow(data, cmap=cmaps.get(name, "gray"), origin="lower", vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
            
        ax = axes[i, -1]
        if plane_name == "YZ":   mask_slice = target_mask[coord, :, :].T
        elif plane_name == "XZ": mask_slice = target_mask[:, coord, :].T
        else:                    mask_slice = target_mask[:, :, coord].T
        im = ax.imshow(mask_slice, cmap="hot", origin="lower", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        
    for ax in axes.flat:
        ax.tick_params(axis="both", which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
    plt.savefig(vis_dir / f"{pdb_id}_fields_slices.png", facecolor="black", dpi=100)
    plt.close(fig)


def update_performance_plot(performance_data: List[Tuple[int, float]], output_path: Path):
    """Updates and saves a scatter plot of processing performance."""
    if not performance_data: return

    atom_counts, times_s = zip(*performance_data)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(atom_counts, times_s, color="cyan", alpha=0.8, edgecolors="lightblue", s=50)
    ax.set_xlabel("Number of Atoms in PQR File", fontsize=14)
    ax.set_ylabel("Execution Time (seconds)", fontsize=14)
    ax.set_title("Feature Generation Performance", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> int:
    """Main function to orchestrate the entire caching process."""
    console.rule("[bold blue]Feature Cache Generation[/bold blue]")
    OUTPUT_ROOT.mkdir(exist_ok=True)

    calculator = SplattingFieldCalculator(resolution=VOXEL_RESOLUTION)
    performance_data = []
    total_start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=NUM_PROCESS_WORKERS) as executor:
        for name, config in DATASET_CONFIGS.items():
            console.rule(f"[bold]Processing Dataset: {name.title()}[/bold]")
            config["cache_dir"].mkdir(parents=True, exist_ok=True)
            config["vis_dir"].mkdir(parents=True, exist_ok=True)

            try:
                df_pdb = pd.read_csv(config["csv_file"], dtype={"pdb_id": str})
                pqr_files = sorted(list(config["pqr_dir"].glob("*.pqr")))
            except FileNotFoundError:
                console.log(f"[yellow]Warning: Could not find input file {config['csv_file']} or directory {config['pqr_dir']}. Skipping dataset.[/yellow]")
                continue

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(), TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Caching [cyan]{name}[/cyan]", total=len(pqr_files))
                
                for pqr_path in pqr_files:
                    pdb_id = pqr_path.stem.upper()
                    cache_file = config["cache_dir"] / f"{pdb_id}.h5"
                    if cache_file.exists() and not FORCE_RECACHE:
                        progress.update(task, advance=1)
                        continue

                    pdb_path = config["pdb_dir"] / f"{pdb_id}.pdb"
                    if not pdb_path.exists():
                        progress.update(task, advance=1)
                        continue

                    try:
                        protein_atoms = calculator.parse_protein_from_pqr(pqr_path)
                        
                        ligand_names = []
                        if "ligands" in df_pdb.columns and pdb_id in df_pdb["pdb_id"].values:
                            ligand_str = df_pdb.loc[df_pdb["pdb_id"] == pdb_id, "ligands"].iloc[0]
                            if pd.notna(ligand_str):
                                ligand_names = [name.strip() for name in ligand_str.split(';')]

                        ligand_atoms = calculator.parse_ligand_from_pdb(pdb_path, ligand_names)
                        
                        start_calc_time = time.perf_counter()
                        results = calculator.calculate_all_fields(protein_atoms, ligand_atoms)
                        calc_time = time.perf_counter() - start_calc_time

                        if results is None:
                            progress.update(task, advance=1)
                            continue

                        features_np, target_mask, num_atoms, grid_params = results
                        executor.submit(
                            process_and_visualize_task, pdb_id, features_np, target_mask,
                            cache_file, config["vis_dir"], ligand_atoms, grid_params)

                        performance_data.append((num_atoms, calc_time))
                        update_performance_plot(performance_data, OUTPUT_ROOT / "performance_plot.png")

                    except Exception as e:
                        console.log(f"[bold red]Error processing {pdb_id}: {e}[/bold red]")
                    
                    progress.update(task, advance=1)

    total_time = time.perf_counter() - total_start_time
    console.rule("[bold green]All Operations Complete[/bold green]")
    console.print(f"Total execution time: {total_time / 60:.2f} min ({total_time:.2f} sec).")
    console.print(f"Cache and visualizations saved in: [cyan]{OUTPUT_ROOT.resolve()}[/cyan]")
    return 0


if __name__ == "__main__":
    sys.exit(main())