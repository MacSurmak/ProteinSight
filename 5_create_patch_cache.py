# -*- coding: utf-8 -*-
"""
Prepares all necessary data caches and splits for U-Net model training.

This script orchestrates the final data preparation stage before training.
It ensures that the training, validation, and test sets are disjoint at the
protein cluster level, preventing data leakage and enabling a more honest
evaluation of the model's generalization capabilities.

The pipeline consists of the following steps:
1.  Loads protein metadata from the positive, negative, and albumin reports.
2.  Splits the unique protein cluster IDs into training, validation, and test
    sets. Albumin clusters are reserved for validation and testing.
3.  Saves the generated splits to a JSON file for use by the training script.
4.  Iterates through all HDF5 feature caches in parallel. For each structure:
    a. Generates and saves a compressed .npz file containing three types of
       coordinates:
         - 'hot_coords': Points within the binding site (target_mask > 0.1).
         - 'surface_coords': Points on the solvent-accessible surface
           (SASA > threshold).
         - 'cold_surface_coords': Surface points guaranteed to be distant
           from any part of the binding site.
    b. Immediately generates and saves a 2D visualization of orthogonal
       slices, illustrating the classified coordinate types.
5.  Provides a final summary of successfully processed and failed files.
"""
import json
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from scipy.ndimage import distance_transform_edt

# Use a non-interactive backend for matplotlib to prevent GUI windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Suppress a common, benign UserWarning from matplotlib when creating many plots
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# --- Configuration ---
console = Console()

# Input directories and files
HDF5_POSITIVE_DIR = Path("feature_cache/positive_cache")
HDF5_NEGATIVE_DIR = Path("feature_cache/negative_cache")
HDF5_ALBUMINS_DIR = Path("feature_cache/albumins_cache")

REPORT_POSITIVE_CSV = Path("protein_clustering_results/clustered_protein_report.csv")
REPORT_NEGATIVE_CSV = Path("negative_controls_results/negative_controls_results_report.csv")
REPORT_ALBUMINS_CSV = Path("albumins_results/albumins_results_report.csv")

# Output directories and files
OUTPUT_ROOT = Path("patch_coordinates_cache")
NPZ_CACHE_DIR = OUTPUT_ROOT / "npz_cache"
VISUALIZATION_DIR = OUTPUT_ROOT / "visualizations"
DATASET_SPLITS_JSON = OUTPUT_ROOT / "dataset_splits.json"

# Parameters
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train/Validation/Test ratios
VOXEL_RESOLUTION = 0.5             # Angstroms per voxel
COLD_DISTANCE_A = 16.0             # Min distance from binding site to be "cold" (Angstroms)
SASA_SURFACE_THRESHOLD = 0.1       # Min SASA value to be considered surface
MAX_WORKERS = 16

# Constants for feature channels
SASA_IDX = 7  # The index of the SASA channel in the feature tensor


def generate_dataset_splits(
    reports: Dict[str, Path],
    output_path: Path,
    split_ratios: Tuple[float, float, float],
    random_seed: int,
) -> None:
    """
    Generates and saves train/validation/test splits based on cluster IDs.
    """
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    cluster_ids = {}
    for name, path in reports.items():
        try:
            df = pd.read_csv(path)
            cluster_ids[name] = sorted(df["cluster_id"].unique().tolist())
        except (FileNotFoundError, KeyError):
            console.log(f"[yellow]Warning: Could not load report {path}. Skipping.[/yellow]")
            cluster_ids[name] = []

    rng = random.Random(random_seed)
    splits = {"train": {}, "validation": {}, "test": {}}
    
    for group in ["positive", "negative"]:
        group_clusters = cluster_ids.get(group, [])
        rng.shuffle(group_clusters)
        n_total, n_train = len(group_clusters), int(len(group_clusters) * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        splits["train"][group] = group_clusters[:n_train]
        splits["validation"][group] = group_clusters[n_train : n_train + n_val]
        splits["test"][group] = group_clusters[n_train + n_val :]

    albumin_clusters = cluster_ids.get("albumins", [])
    splits["validation"]["albumins"] = albumin_clusters
    splits["test"]["albumins"] = albumin_clusters
    splits["train"]["albumins"] = []

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=4)
        
    console.print(f"Dataset splits generated and saved to [cyan]{output_path}[/cyan]")
    console.print(f"  - Train: {len(splits['train']['positive'])} pos / {len(splits['train']['negative'])} neg clusters")
    console.print(f"  - Validation: {len(splits['validation']['positive'])} pos / {len(splits['validation']['negative'])} neg / {len(albumin_clusters)} albumin clusters")


def process_and_visualize_structure(
    task: Tuple[Path, Path, Path, float, float]
) -> Tuple[str, bool, str]:
    """
    Processes a single HDF5 file to cache coordinates and generate a visualization.
    """
    h5_path, npz_dir, vis_dir, cold_dist_voxels, sasa_threshold = task
    pdb_id = h5_path.stem
    npz_path = npz_dir / f"{pdb_id}.npz"

    try:
        # Step 1: Calculate coordinate types
        with h5py.File(h5_path, "r") as f:
            target_mask = f["target_mask"][:]
            sasa = f["features"][SASA_IDX, :, :, :]

        hot_coords = np.argwhere(target_mask > 0.1).astype(np.int16)
        surface_coords = np.argwhere(sasa > sasa_threshold).astype(np.int16)

        if surface_coords.shape[0] == 0:
            return pdb_id, False, "No surface coordinates found (SASA threshold too high?)"

        cold_surface_coords = surface_coords
        if hot_coords.shape[0] > 0:
            hot_mask = np.zeros(target_mask.shape, dtype=bool)
            hot_mask[tuple(hot_coords.T)] = True
            
            # Calculate distance from every voxel to the nearest hot_coord
            dist_transform = distance_transform_edt(~hot_mask)
            
            # Get distances for surface points only
            surface_distances = dist_transform[tuple(surface_coords.T)]
            
            # A cold point's distance must be > cold_dist_voxels
            cold_mask = surface_distances > cold_dist_voxels
            cold_surface_coords = surface_coords[cold_mask]

        # Step 2: Save coordinates to NPZ cache
        np.savez_compressed(
            npz_path,
            hot_coords=hot_coords,
            surface_coords=surface_coords,
            cold_surface_coords=cold_surface_coords,
        )

        # Step 3: Create visualization
        create_visualization(pdb_id, sasa.shape, npz_path, vis_dir)
        
        return pdb_id, True, ""
    except Exception as e:
        return pdb_id, False, str(e)


def create_visualization(
    pdb_id: str, canvas_shape: Tuple, npz_path: Path, output_dir: Path
):
    """Generates a 3-panel visualization of coordinate types."""
    coords = np.load(npz_path)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    
    # Draw layers: Surface -> Cold Surface -> Hot
    canvas[tuple(coords["surface_coords"].T)] = 1
    canvas[tuple(coords["cold_surface_coords"].T)] = 2
    canvas[tuple(coords["hot_coords"].T)] = 3
    
    # Determine slice center
    center_coords = coords["hot_coords"] if coords["hot_coords"].shape[0] > 0 else coords["surface_coords"]
    if center_coords.shape[0] == 0: return
    center = center_coords.mean(axis=0).astype(int)
    cz, cy, cx = np.clip(center, 0, np.array(canvas_shape) - 1)
    
    cmap = ListedColormap(['#101010', '#00BFFF', '#808080', '#FF00FF']) # BG, Surface, Cold, Hot
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    fig.suptitle(f"Coordinate Types for {pdb_id}", color="white", fontsize=16)
    
    axes[0].imshow(canvas[cz, :, :].T, cmap=cmap, norm=norm, origin="lower")
    axes[0].set_title(f"XY Slice (Z={cz})", color="white")
    axes[1].imshow(canvas[:, cy, :].T, cmap=cmap, norm=norm, origin="lower")
    axes[1].set_title(f"XZ Slice (Y={cy})", color="white")
    axes[2].imshow(canvas[:, :, cx].T, cmap=cmap, norm=norm, origin="lower")
    axes[2].set_title(f"YZ Slice (X={cx})", color="white")

    for ax in axes: ax.tick_params(colors="white")
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, 
                        orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'Surface (SASA)', 'Cold Surface', 'Binding Site'])
    cbar.ax.tick_params(colors='white', length=0)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(output_dir / f"{pdb_id}_coords.png", facecolor="black", dpi=120)
    plt.close(fig)


def main():
    """Main function to orchestrate the cache and split generation."""
    console.rule("[bold blue]Step 1: Generating Dataset Splits[/bold blue]")
    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    NPZ_CACHE_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)
    
    reports = {
        "positive": REPORT_POSITIVE_CSV, "negative": REPORT_NEGATIVE_CSV, "albumins": REPORT_ALBUMINS_CSV
    }
    generate_dataset_splits(reports, DATASET_SPLITS_JSON, SPLIT_RATIOS, RANDOM_SEED)

    console.rule("[bold blue]Step 2: Caching Coordinates and Generating Visualizations[/bold blue]")
    h5_files = sorted(
        list(HDF5_POSITIVE_DIR.glob("*.h5")) +
        list(HDF5_NEGATIVE_DIR.glob("*.h5")) +
        list(HDF5_ALBUMINS_DIR.glob("*.h5"))
    )

    if not h5_files:
        console.print("[bold red]Error: No HDF5 caches found. Run feature generation first.[/bold red]")
        return

    cold_dist_voxels = COLD_DISTANCE_A / VOXEL_RESOLUTION
    tasks = [(f, NPZ_CACHE_DIR, VISUALIZATION_DIR, cold_dist_voxels, SASA_SURFACE_THRESHOLD) for f in h5_files]
    
    success_count, failed_tasks = 0, []
    with Progress(
        TextColumn("[progress.description]{task.description}"), BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Processing structures", total=len(tasks))
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for pdb_id, success, msg in executor.map(process_and_visualize_structure, tasks):
                if success:
                    success_count += 1
                else:
                    failed_tasks.append((pdb_id, msg))
                progress.update(task_id, advance=1)

    console.rule("[bold green]Preparation Complete[/bold green]")
    console.print(f"Successfully processed: [green]{success_count}[/green] / {len(h5_files)} proteins.")
    console.print(f"Coordinate cache saved in: [cyan]{NPZ_CACHE_DIR}[/cyan]")
    console.print(f"Visualizations saved in: [cyan]{VISUALIZATION_DIR}[/cyan]")
    
    if failed_tasks:
        console.print(f"\nFailed to process: [red]{len(failed_tasks)}[/red] proteins.")
        console.print("[bold red]Error Summary:[/bold red]")
        for pdb_id, error_msg in sorted(failed_tasks):
            console.print(f" - [bold cyan]{pdb_id}[/bold cyan]: {error_msg}")


if __name__ == "__main__":
    main()