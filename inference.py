# file: inference.py
# -*- coding: utf-8 -*-
"""
Performs end-to-end inference on a single protein structure file (PDB or CIF).
This script is universal and supports both GPU (NVIDIA) and CPU-only execution.

This script provides a self-contained pipeline that takes a protein structure
file as input, automatically performs the necessary PDB to PQR conversion,
generates the 8-channel feature representation, and runs the trained 3D U-Net
model to produce a 3D probability map of potential carotenoid-binding sites.

The pipeline consists of the following automated steps:
1.  Parses command-line arguments for input file, output directory, and model path.
2.  Detects available hardware (GPU/CPU) and selects the appropriate backend.
3.  Creates a temporary directory for intermediate files (.pdb, .pqr).
4.  Converts the input file (CIF or PDB) to a standardized PDB format using gemmi.
5.  Executes the external 'pdb2pqr' tool to generate a PQR file with assigned
    charges and protonation states.
6.  Calculates the 8-channel physicochemical feature fields using either a
    high-performance CUDA-based backend or a compatible NumPy/SciPy backend.
7.  Loads the pre-trained U-Net model.
8.  Performs inference over the entire protein volume using a sliding window
    approach with overlapping 3D patches to ensure smooth predictions.
9.  Saves the final 3D prediction map in the .cube format for visualization
    in molecular viewers (e.g., PyMOL, UCSF Chimera) and generates a 2D slice
    preview image.
10. Automatically cleans up all intermediate files.

Example Usage:
    # For GPU execution (if available)
    python inference.py --input_path /path/to/1A06.pdb --output_dir /path/to/results/1A06

    # To force CPU execution
    python inference.py --input_path /path/to/1A06.pdb --output_dir /path/to/results/1A06 --device cpu
"""

import argparse
import math
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from joblib import Parallel, delayed

import gemmi
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from scipy.ndimage import (
    gaussian_laplace,
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
)

from model import Deeper3DUnetWithDropout

# --- Dynamic GPU/CPU Backend Handling ---
console = Console()
try:
    import cupy as cp
    from numba import cuda
    from numba.core.errors import NumbaPerformanceWarning
    from cupyx.scipy.ndimage import gaussian_laplace as gaussian_laplace_gpu

    GPU_AVAILABLE = True
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except ImportError:
    GPU_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# --- Configuration ---
# --- Model and Data Parameters (Must match training script) ---
DEFAULT_MODEL_PATH = Path("unet_training_output/unet_best_model.pth")
PATCH_SIZE = 64
NUM_INPUT_CHANNELS = 8
INFERENCE_BATCH_SIZE = 4  # Reduced for better CPU compatibility
STRIDE = PATCH_SIZE // 2

# --- PDB2PQR Parameters ---
PDB2PQR_FF = "AMBER"
PDB2PQR_PH = "7.4"

# --- Physicochemical Constants ---
SASA_PROBE_RADIUS_A = 1.4
VOXEL_RESOLUTION = 0.5
CHANNELS = [
    "occupancy",
    "hydrophobicity",
    "electrostatic",
    "aromatic_field",
    "shape_index",
    "hbond_donor",
    "hbond_acceptor",
    "sasa",
]
CHANNEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}
KYTE_DOOLITTLE = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}
AROMATIC_RES_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
}
HBOND_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "TRP": {"NE1"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}
HBOND_ACCEPTORS = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}
ELEC_NORM, SHAPE_NORM, OCC_NORM, HB_D_NORM, HB_A_NORM, AROMA_NORM = (
    0.5,
    0.5,
    3.5,
    0.5,
    0.5,
    0.5,
)


# --- Data Structures & Model Architecture ---
@dataclass(frozen=True)
class Atom:
    coords: np.ndarray
    radius: float
    charge: float = 0.0
    hydrophobicity: float = 0.0
    is_aromatic: bool = False
    is_hbond_donor: bool = False
    is_hbond_acceptor: bool = False


@dataclass(frozen=True)
class GridParameters:
    dims: Tuple[int, int, int]
    origin: np.ndarray
    resolution: float


# --- Base Class for Feature Calculation ---
class BaseFieldCalculator:
    def __init__(self, resolution: float):
        self.resolution = resolution

    @staticmethod
    def parse_pqr(pqr_filepath: Path) -> List[Atom]:
        atoms = []
        with pqr_filepath.open("r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        res, name = line[17:20].strip(), line[12:16].strip()
                        coords = np.array(
                            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        )
                        charge, radius = float(line[54:62]), float(line[62:68])
                        atoms.append(
                            Atom(
                                coords,
                                radius,
                                charge,
                                KYTE_DOOLITTLE.get(res, 0.0),
                                name in AROMATIC_RES_ATOMS.get(res, set()),
                                name in HBOND_DONORS.get(res, set()),
                                name in HBOND_ACCEPTORS.get(res, set()),
                            )
                        )
                    except (ValueError, IndexError):
                        continue
        return atoms

    def _get_grid_params(self, atoms: List[Atom]) -> GridParameters:
        if not atoms:
            raise ValueError("Atom list cannot be empty.")
        coords = np.array([a.coords for a in atoms])
        min_c, max_c = coords.min(axis=0) - 12.0, coords.max(axis=0) + 12.0
        dims = tuple(np.ceil((max_c - min_c) / self.resolution).astype(int))
        return GridParameters(dims, min_c, self.resolution)

    def _normalize_fields(self, features: np.ndarray) -> np.ndarray:
        features[0] = np.tanh(features[0] / OCC_NORM)
        features[1] = (2 * (features[1] - (-4.5)) / 9.0) - 1
        features[2] = np.tanh(features[2] / ELEC_NORM)
        features[3] = np.tanh(features[3] / AROMA_NORM)
        features[4] = np.tanh(features[4] / SHAPE_NORM)
        features[5] = np.tanh(features[5] / HB_D_NORM)
        features[6] = np.tanh(features[6] / HB_A_NORM)
        return np.nan_to_num(features).astype(np.float16)


# --- CPU-only Feature Calculator ---
def _process_atom_chunk(
    coords_chunk,
    radii_chunk,
    charges_chunk,
    hydros_chunk,
    aromatic_chunk,
    donors_chunk,
    acceptors_chunk,
    grid_params,
    resolution,
):
    """Processes a chunk of atoms and returns their contribution. Standalone for pickling."""
    dims_zyx = (grid_params.dims[2], grid_params.dims[1], grid_params.dims[0])
    chunk_features = np.zeros((len(CHANNELS), *dims_zyx), dtype=np.float32)

    for i in range(len(coords_chunk)):
        # Extract properties for the current atom
        coords = coords_chunk[i]
        radius = radii_chunk[i]
        charge = charges_chunk[i]
        hydrophobicity = hydros_chunk[i]
        is_aromatic = aromatic_chunk[i]
        is_hbond_donor = donors_chunk[i]
        is_hbond_acceptor = acceptors_chunk[i]

        # Splatting algorithm: define a local bounding box for updates
        max_cutoff = 12.0
        min_bound = (coords - max_cutoff - grid_params.origin) / resolution
        max_bound = (coords + max_cutoff - grid_params.origin) / resolution

        z_min, y_min, x_min = np.floor(min_bound[[2, 1, 0]]).astype(int)
        z_max, y_max, x_max = np.ceil(max_bound[[2, 1, 0]]).astype(int)

        z_min, y_min, x_min = np.maximum(0, [z_min, y_min, x_min])
        z_max, y_max, x_max = np.minimum(dims_zyx, [z_max, y_max, x_max])

        # Generate coordinate grid for the local region only
        if z_min >= z_max or y_min >= y_max or x_min >= x_max:
            continue

        z_indices, y_indices, x_indices = np.mgrid[
            z_min:z_max, y_min:y_max, x_min:x_max
        ]

        voxel_coords_x = grid_params.origin[0] + (x_indices + 0.5) * resolution
        voxel_coords_y = grid_params.origin[1] + (y_indices + 0.5) * resolution
        voxel_coords_z = grid_params.origin[2] + (z_indices + 0.5) * resolution

        voxel_coords = np.stack([voxel_coords_x, voxel_coords_y, voxel_coords_z])

        dist_sq = np.sum((voxel_coords - coords.reshape(3, 1, 1, 1)) ** 2, axis=0)

        # Calculate atom's contribution and add it to the local feature slice
        local_slice = (
            slice(None),
            slice(z_min, z_max),
            slice(y_min, y_max),
            slice(x_min, x_max),
        )

        sigma_occ = radius * 0.9
        if sigma_occ > 1e-6:
            chunk_features[0][local_slice[1:]] += np.exp(
                -dist_sq / (2.0 * sigma_occ**2)
            )
        chunk_features[1][local_slice[1:]] += hydrophobicity * np.exp(-dist_sq / 8.0)
        chunk_features[2][local_slice[1:]] += charge * np.exp(-dist_sq / 24.5)
        if is_aromatic:
            chunk_features[3][local_slice[1:]] += np.exp(-dist_sq / 2.88)
        if is_hbond_donor:
            chunk_features[5][local_slice[1:]] += np.exp(-dist_sq / 2.88)
        if is_hbond_acceptor:
            chunk_features[6][local_slice[1:]] += np.exp(-dist_sq / 2.88)

    return chunk_features


class FieldCalculatorCPU(BaseFieldCalculator):

    def calculate_fields(
        self, pqr_path: Path
    ) -> Optional[Tuple[np.ndarray, GridParameters]]:
        protein_atoms = self.parse_pqr(pqr_path)
        if not protein_atoms:
            return None

        grid_params = self._get_grid_params(protein_atoms)

        all_coords = np.array([a.coords for a in protein_atoms])
        all_radii = np.array([a.radius for a in protein_atoms])
        all_charges = np.array([a.charge for a in protein_atoms])
        all_hydros = np.array([a.hydrophobicity for a in protein_atoms])
        all_aromatic = np.array([a.is_aromatic for a in protein_atoms])
        all_donors = np.array([a.is_hbond_donor for a in protein_atoms])
        all_acceptors = np.array([a.is_hbond_acceptor for a in protein_atoms])

        # Parallelization: split arrays into chunks for each CPU core
        num_cores = max(1, os.cpu_count() // 2)
        coords_chunks = np.array_split(all_coords, num_cores)
        radii_chunks = np.array_split(all_radii, num_cores)
        charges_chunks = np.array_split(all_charges, num_cores)
        hydros_chunks = np.array_split(all_hydros, num_cores)
        aromatic_chunks = np.array_split(all_aromatic, num_cores)
        donors_chunks = np.array_split(all_donors, num_cores)
        acceptors_chunks = np.array_split(all_acceptors, num_cores)

        console.print(f"  > Calculating features on {num_cores} CPU core(s)...")

        results = Parallel(n_jobs=num_cores)(
            delayed(_process_atom_chunk)(
                coords_chunks[i],
                radii_chunks[i],
                charges_chunks[i],
                hydros_chunks[i],
                aromatic_chunks[i],
                donors_chunks[i],
                acceptors_chunks[i],
                grid_params,
                self.resolution,
            )
            for i in range(num_cores)
        )

        # Sum results from all processes
        features = np.sum(results, axis=0)

        # Shape index and SASA are calculated after summation
        features[CHANNEL_MAP["shape_index"]] = -gaussian_laplace(
            features[0].copy(), sigma=2.0
        )

        # --- SASA calculation ---
        # 1. Create a precise binary mask of the protein based on VdW radii
        protein_mask = np.zeros(grid_params.dims[::-1], dtype=bool)
        for atom in protein_atoms:
            min_bound = np.floor(
                (atom.coords - atom.radius - grid_params.origin) / self.resolution
            ).astype(int)
            max_bound = np.ceil(
                (atom.coords + atom.radius - grid_params.origin) / self.resolution
            ).astype(int)
            z_min, y_min, x_min = np.maximum(0, min_bound[[2, 1, 0]])
            z_max, y_max, x_max = np.minimum(
                grid_params.dims[::-1], max_bound[[2, 1, 0]]
            )

            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                continue

            z_indices, y_indices, x_indices = np.mgrid[
                z_min:z_max, y_min:y_max, x_min:x_max
            ]
            voxel_coords_x = grid_params.origin[0] + (x_indices + 0.5) * self.resolution
            voxel_coords_y = grid_params.origin[1] + (y_indices + 0.5) * self.resolution
            voxel_coords_z = grid_params.origin[2] + (z_indices + 0.5) * self.resolution
            voxel_coords = np.stack([voxel_coords_x, voxel_coords_y, voxel_coords_z])

            dist_sq = np.sum(
                (voxel_coords - atom.coords.reshape(3, 1, 1, 1)) ** 2, axis=0
            )
            protein_mask[z_min:z_max, y_min:y_max, x_min:x_max] |= (
                dist_sq <= atom.radius**2
            )
            dist_sq = np.sum(
                (voxel_coords - atom.coords.reshape(3, 1, 1, 1)) ** 2, axis=0
            )
            protein_mask[z_min:z_max, y_min:y_max, x_min:x_max] |= (
                dist_sq <= atom.radius**2
            )

        # 2. Dilate the mask using a spherical kernel to match GPU logic
        probe_radius_voxels = SASA_PROBE_RADIUS_A / self.resolution
        r_int = int(np.ceil(probe_radius_voxels))
        # Create a grid for the kernel
        kernel_grid = np.mgrid[
            -r_int : r_int + 1, -r_int : r_int + 1, -r_int : r_int + 1
        ]
        # Calculate distance from center and create a spherical boolean mask
        spherical_kernel = np.sum(kernel_grid**2, axis=0) <= probe_radius_voxels**2
        # Perform dilation with the spherical kernel
        dilated_mask = binary_dilation(protein_mask, structure=spherical_kernel)

        # 3. Count accessible neighbors for each voxel on the dilated surface
        sasa_grid = np.zeros_like(features[0], dtype=np.float32)
        padded_dilated = np.pad(
            dilated_mask, pad_width=1, mode="constant", constant_values=0
        )
        # Find coordinates of all surface voxels to iterate over them only
        surface_coords = np.argwhere(dilated_mask)

        for z, y, x in surface_coords:
            # Count occupied neighbors in a 3x3x3 cube in the padded array
            neighborhood = padded_dilated[z : z + 3, y : y + 3, x : x + 3]
            # Accessible neighbors = total (27) - occupied neighbors - self (which is 1)
            # Correct formula: Total neighbors (26) - occupied neighbors (sum-1)
            accessible_neighbors = 26 - (np.sum(neighborhood) - 1)
            sasa_grid[z, y, x] = accessible_neighbors

        # 4. Normalize and assign to the feature channel
        features[CHANNEL_MAP["sasa"]] = sasa_grid / 26.0

        return self._normalize_fields(features), grid_params


# --- GPU Feature Calculator (if available) ---
if GPU_AVAILABLE:

    @cuda.jit
    def splatting_kernel(
        out,
        grid_min,
        grid_dims,
        res,
        coords,
        charges,
        hydros,
        donor_flags,
        acceptor_flags,
        aromatic_flags,
        atom_radii,
    ):
        i = cuda.grid(1)
        if i >= coords.shape[0]:
            return
        radius, sigma_occ, max_cutoff = atom_radii[i], atom_radii[i] * 0.9, 12.0
        sigma_sq_occ = sigma_occ * sigma_occ
        pos_x, pos_y, pos_z = coords[i, 0], coords[i, 1], coords[i, 2]
        min_x, max_x = max(0, int(((pos_x - max_cutoff) - grid_min[0]) / res)), min(
            grid_dims[0], int(((pos_x + max_cutoff) - grid_min[0]) / res) + 1
        )
        min_y, max_y = max(0, int(((pos_y - max_cutoff) - grid_min[1]) / res)), min(
            grid_dims[1], int(((pos_y + max_cutoff) - grid_min[1]) / res) + 1
        )
        min_z, max_z = max(0, int(((pos_z - max_cutoff) - grid_min[2]) / res)), min(
            grid_dims[2], int(((pos_z + max_cutoff) - grid_min[2]) / res) + 1
        )
        for z_idx in range(min_z, max_z):
            for y_idx in range(min_y, max_y):
                for x_idx in range(min_x, max_x):
                    v_pos_x, v_pos_y, v_pos_z = (
                        grid_min[0] + (x_idx + 0.5) * res,
                        grid_min[1] + (y_idx + 0.5) * res,
                        grid_min[2] + (z_idx + 0.5) * res,
                    )
                    dist_sq = (
                        (v_pos_x - pos_x) ** 2
                        + (v_pos_y - pos_y) ** 2
                        + (v_pos_z - pos_z) ** 2
                    )
                    if dist_sq < (4 * sigma_occ) ** 2:
                        cuda.atomic.add(
                            out,
                            (0, z_idx, y_idx, x_idx),
                            math.exp(-dist_sq / (2.0 * sigma_sq_occ)),
                        )
                    if dist_sq < 64.0:
                        cuda.atomic.add(
                            out,
                            (1, z_idx, y_idx, x_idx),
                            hydros[i] * math.exp(-dist_sq / 8.0),
                        )
                    if dist_sq < 144.0:
                        cuda.atomic.add(
                            out,
                            (2, z_idx, y_idx, x_idx),
                            charges[i] * math.exp(-dist_sq / 24.5),
                        )
                    if aromatic_flags[i] and dist_sq < 25.0:
                        cuda.atomic.add(
                            out, (3, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88)
                        )
                    if donor_flags[i] and dist_sq < 25.0:
                        cuda.atomic.add(
                            out, (5, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88)
                        )
                    if acceptor_flags[i] and dist_sq < 25.0:
                        cuda.atomic.add(
                            out, (6, z_idx, y_idx, x_idx), math.exp(-dist_sq / 2.88)
                        )

    @cuda.jit
    def splatting_vdw_mask_kernel(
        mask_out, grid_min, grid_dims, res, atom_coords, atom_radii
    ):
        i = cuda.grid(1)
        if i >= atom_coords.shape[0]:
            return
        radius, radius_sq = atom_radii[i], atom_radii[i] ** 2
        pos_x, pos_y, pos_z = atom_coords[i, 0], atom_coords[i, 1], atom_coords[i, 2]
        min_x, max_x = max(0, int(((pos_x - radius) - grid_min[0]) / res)), min(
            grid_dims[0], int(((pos_x + radius) - grid_min[0]) / res) + 1
        )
        min_y, max_y = max(0, int(((pos_y - radius) - grid_min[1]) / res)), min(
            grid_dims[1], int(((pos_y + radius) - grid_min[1]) / res) + 1
        )
        min_z, max_z = max(0, int(((pos_z - radius) - grid_min[2]) / res)), min(
            grid_dims[2], int(((pos_z + radius) - grid_min[2]) / res) + 1
        )
        for z_idx in range(min_z, max_z):
            for y_idx in range(min_y, max_y):
                for x_idx in range(min_x, max_x):
                    v_pos_x, v_pos_y, v_pos_z = (
                        grid_min[0] + (x_idx + 0.5) * res,
                        grid_min[1] + (y_idx + 0.5) * res,
                        grid_min[2] + (z_idx + 0.5) * res,
                    )
                    if (
                        (v_pos_x - pos_x) ** 2
                        + (v_pos_y - pos_y) ** 2
                        + (v_pos_z - pos_z) ** 2
                    ) <= radius_sq:
                        cuda.atomic.max(mask_out, (z_idx, y_idx, x_idx), 1.0)

    @cuda.jit
    def splatting_dilation_kernel(
        dilated_mask_out, protein_mask_in, probe_radius_voxels
    ):
        z, y, x = cuda.grid(3)
        dims = protein_mask_in.shape
        if (
            z >= dims[0]
            or y >= dims[1]
            or x >= dims[2]
            or protein_mask_in[z, y, x] == 0
        ):
            return
        r_int, r_sq = (
            int(probe_radius_voxels),
            probe_radius_voxels * probe_radius_voxels,
        )
        for dz in range(-r_int, r_int + 1):
            for dy in range(-r_int, r_int + 1):
                for dx in range(-r_int, r_int + 1):
                    if (dx * dx + dy * dy + dz * dz) > r_sq:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < dims[0] and 0 <= ny < dims[1] and 0 <= nx < dims[2]:
                        cuda.atomic.max(dilated_mask_out, (nz, ny, nx), 1.0)

    @cuda.jit
    def calculate_sasa_kernel(sasa_out, protein_mask_in):
        z, y, x = cuda.grid(3)
        dims = protein_mask_in.shape
        if (
            z >= dims[0]
            or y >= dims[1]
            or x >= dims[2]
            or protein_mask_in[z, y, x] == 0
        ):
            return
        accessible_neighbors = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (
                        not (
                            0 <= nz < dims[0]
                            and 0 <= ny < dims[1]
                            and 0 <= nx < dims[2]
                        )
                        or protein_mask_in[nz, ny, nx] == 0
                    ):
                        accessible_neighbors += 1
        sasa_out[z, y, x] = accessible_neighbors

    class FieldCalculatorGPU(BaseFieldCalculator):
        def calculate_fields(
            self, pqr_path: Path
        ) -> Optional[Tuple[np.ndarray, GridParameters]]:
            protein_atoms = self.parse_pqr(pqr_path)
            if not protein_atoms:
                return None

            num_atoms = len(protein_atoms)
            grid_params = self._get_grid_params(protein_atoms)
            dims_zyx = (grid_params.dims[2], grid_params.dims[1], grid_params.dims[0])
            grid_min_d = cp.asarray(grid_params.origin, dtype=cp.float32)
            grid_dims_d = cp.asarray(grid_params.dims, dtype=cp.int32)

            prot_coords_d = cp.asarray(
                [a.coords for a in protein_atoms], dtype=cp.float32
            )
            prot_radii_d = cp.asarray(
                [a.radius for a in protein_atoms], dtype=cp.float32
            )

            protein_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
            grid_1d = (num_atoms + 256 - 1) // 256
            splatting_vdw_mask_kernel[grid_1d, 256](
                protein_mask_d,
                grid_min_d,
                grid_dims_d,
                self.resolution,
                prot_coords_d,
                prot_radii_d,
            )

            features_d = cp.zeros((len(CHANNELS), *dims_zyx), dtype=cp.float32)
            charges_d = cp.asarray([a.charge for a in protein_atoms], dtype=cp.float32)
            hydros_d = cp.asarray(
                [a.hydrophobicity for a in protein_atoms], dtype=cp.float32
            )
            donor_d = cp.asarray(
                [a.is_hbond_donor for a in protein_atoms], dtype=cp.bool_
            )
            acceptor_d = cp.asarray(
                [a.is_hbond_acceptor for a in protein_atoms], dtype=cp.bool_
            )
            aromatic_d = cp.asarray(
                [a.is_aromatic for a in protein_atoms], dtype=cp.bool_
            )

            splatting_kernel[grid_1d, 256](
                features_d,
                grid_min_d,
                grid_dims_d,
                self.resolution,
                prot_coords_d,
                charges_d,
                hydros_d,
                donor_d,
                acceptor_d,
                aromatic_d,
                prot_radii_d,
            )

            features_d[CHANNEL_MAP["shape_index"]] = -gaussian_laplace_gpu(
                features_d[0].copy(), sigma=2.0
            )

            probe_radius_voxels = SASA_PROBE_RADIUS_A / self.resolution
            expanded_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
            grid_3d = tuple((s + b - 1) // b for s, b in zip(dims_zyx, (8, 8, 8)))
            splatting_dilation_kernel[grid_3d, (8, 8, 8)](
                expanded_mask_d, protein_mask_d, probe_radius_voxels
            )
            sasa_d = cp.zeros(dims_zyx, dtype=cp.float32)
            calculate_sasa_kernel[grid_3d, (8, 8, 8)](sasa_d, expanded_mask_d)
            features_d[CHANNEL_MAP["sasa"]] = sasa_d / 26.0

            final_features = self._normalize_fields(features_d)
            final_features_np = cp.asnumpy(final_features)
            cp.get_default_memory_pool().free_all_blocks()
            return final_features_np, grid_params


# --- Utility Functions (PQR prep, inference, saving results) ---
def prepare_pqr_file(input_path: Path, temp_dir: Path) -> Path:
    pdb_id = input_path.stem
    temp_pdb_path = temp_dir / f"{pdb_id}.pdb"
    temp_pqr_path = temp_dir / f"{pdb_id}.pqr"

    try:
        if input_path.suffix.lower() == ".cif":
            structure = gemmi.read_structure(str(input_path))
            structure.write_pdb(str(temp_pdb_path), ter_records=False)
        elif input_path.suffix.lower() == ".pdb":
            import shutil

            shutil.copy(input_path, temp_pdb_path)
        else:
            raise ValueError(f"Unsupported input file format: {input_path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to process input file with gemmi: {e}") from e

    command = [
        "pdb2pqr",
        f"--ff={PDB2PQR_FF}",
        f"--with-ph={PDB2PQR_PH}",
        "--drop-water",
        str(temp_pdb_path),
        str(temp_pqr_path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "'pdb2pqr' command not found. Please ensure it is installed and in your system's PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pdb2pqr failed with error:\n{e.stderr.strip()}") from e

    return temp_pqr_path


def run_sliding_window_inference(
    model: nn.Module, input_tensor: torch.Tensor, device: str
) -> np.ndarray:
    model.eval()
    C, D, H, W = input_tensor.shape
    pad_D = (STRIDE - (D - PATCH_SIZE) % STRIDE) % STRIDE
    pad_H = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_W = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE
    padded_input = nn.functional.pad(input_tensor, (0, pad_W, 0, pad_H, 0, pad_D))
    pD, pH, pW = padded_input.shape[1:]
    prediction_map = torch.zeros((1, pD, pH, pW), device=device, dtype=torch.float32)
    weight_map = torch.zeros_like(prediction_map)
    grid = torch.arange(0, PATCH_SIZE, device=device, dtype=torch.float32)
    center = (PATCH_SIZE - 1) / 2.0
    gaussian_1d = torch.exp(-((grid - center) ** 2 / (2 * (PATCH_SIZE / 4.0) ** 2)))
    gaussian_weights = (
        torch.outer(torch.outer(gaussian_1d, gaussian_1d).flatten(), gaussian_1d)
        .reshape(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
        .unsqueeze(0)
        .to(device)
    )
    patch_coords = [
        (z, y, x)
        for z in range(0, pD - PATCH_SIZE + 1, STRIDE)
        for y in range(0, pH - PATCH_SIZE + 1, STRIDE)
        for x in range(0, pW - PATCH_SIZE + 1, STRIDE)
    ]

    for i in range(0, len(patch_coords), INFERENCE_BATCH_SIZE):
        batch_info = patch_coords[i : i + INFERENCE_BATCH_SIZE]
        patches = [
            padded_input[:, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            for z, y, x in batch_info
        ]
        batch_tensor = torch.stack(patches)
        if device == "cuda":
            batch_tensor = batch_tensor.to(memory_format=torch.channels_last_3d)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = torch.sigmoid(model(batch_tensor))
        for j, (z, y, x) in enumerate(batch_info):
            prediction_map[
                :, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE
            ] += (outputs[j] * gaussian_weights)
            weight_map[
                :, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE
            ] += gaussian_weights

    final_prediction = prediction_map / torch.clamp(weight_map, min=1e-8)
    return final_prediction.squeeze(0).cpu().numpy()[:D, :H, :W]


def save_prediction_as_cube(
    prediction: np.ndarray, grid_params: GridParameters, output_path: Path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = prediction.shape
    prediction_transposed = prediction.transpose(2, 1, 0)
    origin_bohr = grid_params.origin * (1 / 0.529177)
    res_bohr = grid_params.resolution * (1 / 0.529177)
    with output_path.open("w") as f:
        f.write("U-Net Prediction for Binding Site\nGenerated by inference.py\n")
        f.write(
            f"    1 {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n"
        )
        f.write(f"{nx:5d} {res_bohr:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {res_bohr:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {res_bohr:12.6f}\n")
        f.write(f"    1   0.00000    {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        for i, val in enumerate(prediction_transposed.flatten()):
            f.write(f" {val:13.5e}")
            f.write("\n") if (i + 1) % 6 == 0 else None
        if prediction.size % 6 != 0:
            f.write("\n")


def visualize_inference_results(
    output_dir: Path, pdb_id: str, features: np.ndarray, prediction: np.ndarray
):
    output_path = output_dir / f"{pdb_id}_inference_slice.png"
    center = (np.array(prediction.shape) / 2).astype(int)
    cz, cy, cx = center
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="black")
    fig.suptitle(
        f"Inference Result for {pdb_id} (Central XY Slice)", color="white", fontsize=16
    )
    features_f32 = features.astype(np.float32)
    rgb_input = np.stack(
        [
            features_f32[CHANNEL_MAP["sasa"]][cz, :, :],
            (features_f32[CHANNEL_MAP["hydrophobicity"]][cz, :, :] + 1) / 2,
            (features_f32[CHANNEL_MAP["electrostatic"]][cz, :, :] + 1) / 2,
        ],
        axis=-1,
    )
    axes[0].imshow(np.clip(rgb_input, 0, 1), origin="lower")
    axes[0].set_title("Input Features (R:SASA, G:Hyd, B:Elec)", color="white")
    im = axes[1].imshow(
        prediction[cz, :, :], cmap="hot", origin="lower", vmin=0, vmax=1
    )
    axes[1].set_title("Model Prediction", color="white")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, facecolor="black", dpi=150)
    plt.close(fig)


def main(args):
    """Main execution function for the inference pipeline."""
    pdb_id = args.input_path.stem
    console.rule(f"[bold blue]Starting Inference for {pdb_id}[/bold blue]")

    # --- Step 0: Setup Backend ---
    use_gpu = GPU_AVAILABLE and (args.device == "cuda")
    device = "cuda" if use_gpu else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        console.print(
            "[yellow]Warning: CUDA device requested, but not available. Falling back to CPU.[/yellow]"
        )
        device = "cpu"
        use_gpu = False

    console.print(
        f"Using [bold]{'GPU (CUDA)' if use_gpu else 'CPU'}[/bold] backend for computation."
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # --- Step 1: Prepare PQR file ---
        console.print(
            f"[1/5] Preparing PQR file from [cyan]{args.input_path.name}[/cyan]..."
        )
        start_time = time.perf_counter()
        try:
            pqr_path = prepare_pqr_file(args.input_path, temp_dir)
            console.print(
                f"  > PQR preparation time: {time.perf_counter() - start_time:.2f} s"
            )
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            console.print(f"[bold red]Error during PQR preparation: {e}[/bold red]")
            sys.exit(1)

        # --- Step 2: Calculate feature fields ---
        console.print("[2/5] Calculating 8-channel feature fields...")
        start_time = time.perf_counter()

        CalculatorClass = FieldCalculatorGPU if use_gpu else FieldCalculatorCPU
        calculator = CalculatorClass(resolution=VOXEL_RESOLUTION)

        results = calculator.calculate_fields(pqr_path)
        if results is None:
            console.print(
                f"[bold red]Error: Failed to calculate feature fields.[/bold red]"
            )
            sys.exit(1)
        features_np, grid_params = results
        console.print(
            f"  > Field calculation time: {time.perf_counter() - start_time:.2f} s"
        )
        console.print(f"  > Grid dimensions: {features_np.shape}")

        # --- Step 3: Load Model ---
        console.print(
            f"[3/5] Loading U-Net model from [cyan]{args.model_path.name}[/cyan]..."
        )
        model = Deeper3DUnetWithDropout(in_channels=NUM_INPUT_CHANNELS, out_channels=1)
        if not args.model_path.exists():
            console.print(
                f"[bold red]Error: Model file not found at {args.model_path}[/bold red]"
            )
            sys.exit(1)

        model.to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        if use_gpu:
            model = model.to(memory_format=torch.channels_last_3d)
            model = torch.compile(model)
        console.print(f"  > Model loaded onto [bold]{device}[/bold].")

        # --- Step 4: Run Inference ---
        console.print("[4/5] Running sliding window inference...")
        start_time = time.perf_counter()
        input_tensor = torch.from_numpy(features_np).to(torch.float32).to(device)
        prediction_np = run_sliding_window_inference(model, input_tensor, device)
        console.print(f"  > Inference time: {time.perf_counter() - start_time:.2f} s")

        # --- Step 5: Save Results ---
        console.print("[5/5] Saving and visualizing results...")
        cube_path = args.output_dir / f"{pdb_id}_prediction.cube"
        save_prediction_as_cube(prediction_np, grid_params, cube_path)
        console.print(f"  > 3D prediction map saved to: [cyan]{cube_path}[/cyan]")
        visualize_inference_results(args.output_dir, pdb_id, features_np, prediction_np)
        console.print(
            f"  > 2D visualization saved to: [cyan]{args.output_dir / f'{pdb_id}_inference_slice.png'}[/cyan]"
        )

    console.rule(f"[bold green]Inference for {pdb_id} complete![/bold green]")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ProteinSight inference on a single protein structure file (PDB or CIF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input protein structure file (.pdb or .cif).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the output prediction.cube and visualization.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the pre-trained model weights file (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Preferred device for inference. Will fall back to CPU if CUDA is not available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
