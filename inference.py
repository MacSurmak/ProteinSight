# -*- coding: utf-8 -*-
"""
Performs inference on a single protein using a trained 3D U-Net model.

This script executes the complete inference pipeline for a new, unseen protein
structure. It employs a sliding window strategy to process proteins of any
size and generates a 3D probability map of potential binding sites.

The pipeline consists of the following steps:
1.  Accepts a PDB ID as a command-line argument.
2.  Locates the corresponding PQR file for the target protein.
3.  Calculates the 8-channel physicochemical feature fields using the same
    high-performance CUDA-based "splatting" methodology as the training
    pipeline, ensuring consistency.
4.  Initializes the U-Net model architecture and loads the pre-trained weights.
5.  Performs inference over the entire protein volume using a sliding window
    approach with overlapping 3D patches.
6.  Reconstructs a smooth, final prediction map by performing a weighted
    average of the patch predictions using a Gaussian weight map to eliminate
    seam artifacts.
7.  Saves the final 3D prediction map in the .cube format for visualization
    in molecular viewers like PyMOL and generates 2D orthogonal slices for
    a quick preview of the results.
"""

import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from tqdm import tqdm

from model import Deeper3DUnetWithDropout

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_laplace as gaussian_laplace_gpu
    from numba import cuda
    from numba.core.errors import NumbaPerformanceWarning
except ImportError:
    sys.exit(
        "Error: Required GPU libraries not found. "
        "Please install: pip install torch numba cupy-cuda12x pandas matplotlib scipy rich tqdm"
    )

# Use a non-interactive backend for matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress expected performance warnings for cleaner output
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


# --- Configuration ---
console = Console()

# --- INPUT: Specify paths to data and trained model ---
# This script is configured to run inference on the albumin dataset by default.
# To run on other datasets, change this path.
PQR_FILES_DIR = Path("positive_pqr_files")
MODEL_SAVE_PATH = Path("unet_training_output/unet_best_model.pth")
OUTPUT_DIR = Path("inference_results")

# --- Model and Data Parameters (Must match training script) ---
PATCH_SIZE = 64
NUM_INPUT_CHANNELS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference Parameters ---
INFERENCE_BATCH_SIZE = 8  # Adjust based on available GPU memory
STRIDE = PATCH_SIZE // 2  # Overlap between patches; //2 is standard

# --- Physicochemical Constants (Synced with 4_prepare_feature_cache.py) ---
SASA_PROBE_RADIUS_A = 1.4
VOXEL_RESOLUTION = 0.5
CUDA_BLOCK_SIZE_1D = 256
CUDA_BLOCK_SIZE_3D = (8, 8, 8)
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
(ELEC_NORM, SHAPE_NORM, OCC_NORM, HB_D_NORM, HB_A_NORM, AROMA_NORM) = (
    0.5,
    0.5,
    3.5,
    0.5,
    0.5,
    0.5,
)


# --- Data Structures and Model Architecture (Synced with 6_train_unet.py) ---
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


# --- CUDA Kernels (Synced with 4_prepare_feature_cache.py) ---
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
    min_x = max(0, int(((pos_x - radius) - grid_min[0]) / res))
    max_x = min(grid_dims[0], int(((pos_x + radius) - grid_min[0]) / res) + 1)
    min_y = max(0, int(((pos_y - radius) - grid_min[1]) / res))
    max_y = min(grid_dims[1], int(((pos_y + radius) - grid_min[1]) / res) + 1)
    min_z = max(0, int(((pos_z - radius) - grid_min[2]) / res))
    max_z = min(grid_dims[2], int(((pos_z + radius) - grid_min[2]) / res) + 1)
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
def splatting_dilation_kernel(dilated_mask_out, protein_mask_in, probe_radius_voxels):
    z, y, x = cuda.grid(3)
    dims = protein_mask_in.shape
    if z >= dims[0] or y >= dims[1] or x >= dims[2] or protein_mask_in[z, y, x] == 0:
        return
    r_int = int(probe_radius_voxels)
    r_sq = probe_radius_voxels * probe_radius_voxels
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
    if z >= dims[0] or y >= dims[1] or x >= dims[2] or protein_mask_in[z, y, x] == 0:
        return
    accessible_neighbors = 0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nz, ny, nx = z + dz, y + dy, x + dx
                if (
                    not (0 <= nz < dims[0] and 0 <= ny < dims[1] and 0 <= nx < dims[2])
                    or protein_mask_in[nz, ny, nx] == 0
                ):
                    accessible_neighbors += 1
    sasa_out[z, y, x] = accessible_neighbors


# --- Main Logic ---
class FieldCalculator:
    """Calculates 8-channel feature fields, synced with the training pipeline."""

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

    def calculate_fields(
        self, pqr_path: Path
    ) -> Optional[Tuple[np.ndarray, GridParameters]]:
        protein_atoms = self.parse_pqr(pqr_path)
        if not protein_atoms:
            return None

        num_atoms = len(protein_atoms)
        grid_params = self._get_grid_params(protein_atoms)
        dims_zyx = (grid_params.dims[2], grid_params.dims[1], grid_params.dims[0])
        grid_min_d, grid_dims_d = cp.asarray(grid_params.origin), cp.asarray(
            grid_params.dims, dtype=cp.int32
        )

        protein_mask_d = cp.zeros(dims_zyx, dtype=cp.float32)
        prot_coords_d = cp.asarray([a.coords for a in protein_atoms])
        prot_radii_d = cp.asarray([a.radius for a in protein_atoms])

        grid_1d = (num_atoms + CUDA_BLOCK_SIZE_1D - 1) // CUDA_BLOCK_SIZE_1D
        splatting_vdw_mask_kernel[grid_1d, CUDA_BLOCK_SIZE_1D](
            protein_mask_d,
            grid_min_d,
            grid_dims_d,
            self.resolution,
            prot_coords_d,
            prot_radii_d,
        )

        features_d = cp.zeros((len(CHANNELS), *dims_zyx), dtype=cp.float32)
        charges_d, hydros_d = cp.asarray([a.charge for a in protein_atoms]), cp.asarray(
            [a.hydrophobicity for a in protein_atoms]
        )
        donor_d, acceptor_d, aromatic_d = (
            cp.asarray([a.is_hbond_donor for a in protein_atoms]),
            cp.asarray([a.is_hbond_acceptor for a in protein_atoms]),
            cp.asarray([a.is_aromatic for a in protein_atoms]),
        )

        splatting_kernel[grid_1d, CUDA_BLOCK_SIZE_1D](
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
        grid_3d = tuple((s + b - 1) // b for s, b in zip(dims_zyx, CUDA_BLOCK_SIZE_3D))
        splatting_dilation_kernel[grid_3d, CUDA_BLOCK_SIZE_3D](
            expanded_mask_d, protein_mask_d, probe_radius_voxels
        )
        sasa_d = cp.zeros(dims_zyx, dtype=cp.float32)
        calculate_sasa_kernel[grid_3d, CUDA_BLOCK_SIZE_3D](sasa_d, expanded_mask_d)
        features_d[CHANNEL_MAP["sasa"]] = sasa_d / 26.0

        features_d[0] = cp.tanh(features_d[0] / OCC_NORM)
        features_d[1] = (2 * (features_d[1] - (-4.5)) / 9.0) - 1
        features_d[2] = cp.tanh(features_d[2] / ELEC_NORM)
        features_d[3] = cp.tanh(features_d[3] / AROMA_NORM)
        features_d[4] = cp.tanh(features_d[4] / SHAPE_NORM)
        features_d[5] = cp.tanh(features_d[5] / HB_D_NORM)
        features_d[6] = cp.tanh(features_d[6] / HB_A_NORM)

        final_features = cp.nan_to_num(features_d).astype(cp.float16).get()
        cp.get_default_memory_pool().free_all_blocks()
        return final_features, grid_params


def run_sliding_window_inference(
    model: nn.Module, input_tensor: torch.Tensor
) -> np.ndarray:
    model.eval()
    C, D, H, W = input_tensor.shape
    pad_D = (STRIDE - (D - PATCH_SIZE) % STRIDE) % STRIDE
    pad_H = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_W = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE

    padded_input = nn.functional.pad(input_tensor, (0, pad_W, 0, pad_H, 0, pad_D))
    pD, pH, pW = padded_input.shape[1:]

    prediction_map = torch.zeros((1, pD, pH, pW), device=DEVICE, dtype=torch.float32)
    weight_map = torch.zeros_like(prediction_map)

    grid = torch.arange(0, PATCH_SIZE, device=DEVICE, dtype=torch.float32)
    center = (PATCH_SIZE - 1) / 2.0
    gaussian_1d = torch.exp(-((grid - center) ** 2 / (2 * (PATCH_SIZE / 4.0) ** 2)))
    gaussian_weights = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_weights = torch.outer(gaussian_weights.flatten(), gaussian_1d).reshape(
        PATCH_SIZE, PATCH_SIZE, PATCH_SIZE
    )
    gaussian_weights = gaussian_weights.unsqueeze(0).to(DEVICE)

    patch_coords = [
        (z, y, x)
        for z in range(0, pD - PATCH_SIZE + 1, STRIDE)
        for y in range(0, pH - PATCH_SIZE + 1, STRIDE)
        for x in range(0, pW - PATCH_SIZE + 1, STRIDE)
    ]

    for i in tqdm(
        range(0, len(patch_coords), INFERENCE_BATCH_SIZE), desc="  Inferring patches"
    ):
        batch_info = patch_coords[i : i + INFERENCE_BATCH_SIZE]
        patches = [
            padded_input[:, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            for z, y, x in batch_info
        ]
        batch_tensor = torch.stack(patches).to(memory_format=torch.channels_last_3d)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            outputs = torch.sigmoid(model(batch_tensor))

        for j, (z, y, x) in enumerate(batch_info):
            prediction_map[
                :, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE
            ] += (outputs[j] * gaussian_weights)
            weight_map[
                :, z : z + PATCH_SIZE, y : y + PATCH_SIZE, x : x + PATCH_SIZE
            ] += gaussian_weights

    final_prediction = prediction_map / torch.clamp(weight_map, min=1e-8)
    return final_prediction.squeeze(0).cpu().numpy(), (pad_D, pad_H, pad_W)


def save_prediction_as_cube(
    prediction: np.ndarray, grid_params: GridParameters, output_path: Path
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = prediction.shape
    prediction_transposed = prediction.transpose(2, 1, 0)
    origin_bohr = grid_params.origin * (1 / 0.529177)
    res_bohr = grid_params.resolution * (1 / 0.529177)

    with output_path.open("w") as f:
        f.write("U-Net Prediction for Binding Site\nGenerated by predict.py\n")
        f.write(
            f"    1 {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n"
        )
        f.write(f"{nx:5d} {res_bohr:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {res_bohr:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {res_bohr:12.6f}\n")
        f.write(f"    1   0.00000    {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        for i, val in enumerate(prediction_transposed.flatten()):
            f.write(f" {val:13.5e}")
            if (i + 1) % 6 == 0:
                f.write("\n")
        if prediction.size % 6 != 0:
            f.write("\n")


def visualize_inference_results(
    pdb_id: str, features: np.ndarray, prediction: np.ndarray
):
    output_path = OUTPUT_DIR / pdb_id / f"{pdb_id}_inference_slices.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    center = (np.array(prediction.shape) / 2).astype(int)
    cz, cy, cx = center

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="black")
    fig.suptitle(
        f"Inference Result for {pdb_id} (XY Slice)", color="white", fontsize=16
    )

    features_f32 = features.astype(np.float32)

    rgb_input = np.stack([
        features_f32[CHANNEL_MAP["sasa"]][cz, :, :],
        (features_f32[CHANNEL_MAP["hydrophobicity"]][cz, :, :] + 1) / 2,
        (features_f32[CHANNEL_MAP["electrostatic"]][cz, :, :] + 1) / 2,
    ], axis=-1)

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
    plt.savefig(output_path, facecolor="black", dpi=120)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        console.print(
            "[bold red]Error: PDB ID must be provided as a command-line argument.[/bold red]"
        )
        console.print(f"Usage: python {sys.argv[0]} <PDB_ID>")
        sys.exit(1)

    pdb_id = sys.argv[1].upper()
    console.rule(f"[bold blue]Starting Inference for {pdb_id}[/bold blue]")
    OUTPUT_DIR.mkdir(exist_ok=True)
    pqr_path = PQR_FILES_DIR / f"{pdb_id}.pqr"

    if not pqr_path.exists():
        console.print(f"[bold red]Error: PQR file not found at {pqr_path}[/bold red]")
        sys.exit(1)
    if not MODEL_SAVE_PATH.exists():
        console.print(
            f"[bold red]Error: Trained model not found at {MODEL_SAVE_PATH}[/bold red]"
        )
        sys.exit(1)

    console.print("\n[1/4] Calculating 8-channel feature fields...")
    start_time = time.time()
    calculator = FieldCalculator(resolution=VOXEL_RESOLUTION)
    results = calculator.calculate_fields(pqr_path)
    if results is None:
        console.print(
            f"[bold red]Error: Failed to calculate feature fields for {pdb_id}.[/bold red]"
        )
        sys.exit(1)

    features_np, grid_params = results
    console.print(f"  > Field calculation time: {time.time() - start_time:.2f} s")
    console.print(f"  > Grid dimensions: {features_np.shape}")

    console.print("\n[2/4] Loading and compiling U-Net model...")
    model = Deeper3DUnetWithDropout(in_channels=NUM_INPUT_CHANNELS, out_channels=1)
    if DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")
        model.to(DEVICE, memory_format=torch.channels_last_3d)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model = torch.compile(model)
    else:
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    console.print(f"  > Model loaded onto {DEVICE} and compiled.")

    console.print("\n[3/4] Running sliding window inference...")
    start_time = time.time()
    input_tensor = torch.from_numpy(features_np).to(torch.float32).to(DEVICE)
    prediction_np = run_sliding_window_inference(model, input_tensor)
    console.print(f"  > Inference time: {time.time() - start_time:.2f} s")

    console.print("\n[4/4] Saving and visualizing results...")
    result_dir = OUTPUT_DIR / pdb_id
    result_dir.mkdir(exist_ok=True)
    cube_path = result_dir / f"{pdb_id}_prediction.cube"
    save_prediction_as_cube(prediction_np, grid_params, cube_path)
    console.print(f"  > 3D prediction map saved to: [cyan]{cube_path}[/cyan]")
    visualize_inference_results(pdb_id, features_np, prediction_np)
    console.print(f"  > 2D visualization saved in: [cyan]{result_dir}[/cyan]")

    console.rule(f"[bold green]Inference for {pdb_id} complete![/bold green]")


if __name__ == "__main__":
    main()
