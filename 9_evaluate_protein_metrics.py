# -*- coding: utf-8 -*-
"""
Evaluates the model by classifying islands and calculating their volume ratios.

This script provides a nuanced, fragmentation-resistant evaluation by:
1.  Identifying all discrete "islands" in both the ground truth and prediction masks.
2.  Classifying each ground truth island as "found" or "missed" based on
    whether it overlaps with any prediction island.
3.  Classifying each prediction island as "true positive" or "false positive"
    based on overlap with the ground truth.
4.  Aggregating the voxel volumes of these categories to calculate robust,
    volume-based recall and precision metrics. This focuses on how much of the
    true signal volume was captured, rather than penalizing for segmentation shape.
"""
import warnings
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy.ndimage import label

from inference import run_sliding_window_inference
from model import Deeper3DUnetWithDropout

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Configuration ---
console = Console()
HDF5_CACHE_ROOT = Path("feature_cache")
DATASET_SPLIT_REPORT_CSV = Path("dataset_split_report.csv")
MODEL_SAVE_PATH = Path("unet_training_output/unet_best_model.pth")
OUTPUT_DIR = Path("evaluation_results_island_volume")
METRICS_CSV_PATH = OUTPUT_DIR / "island_volume_metrics_per_protein.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INPUT_CHANNELS = 8
PREDICTION_THRESHOLD = 0.5
GT_THRESHOLD = 1e-6
MIN_FP_VOXEL_COUNT = 20


def calculate_island_volume_metrics(
    prediction_map: np.ndarray, ground_truth_mask: np.ndarray
) -> Dict[str, float]:
    """
    Classifies each island and calculates the total volume for each category.
    """
    pred_binary_mask = prediction_map > PREDICTION_THRESHOLD
    pred_labeled, n_pred_islands = label(pred_binary_mask)

    gt_binary_mask = ground_truth_mask > GT_THRESHOLD
    gt_labeled, n_gt_sites = label(gt_binary_mask)

    # Initialize volumes
    v_found_gt = 0
    v_missed_gt = 0
    v_tp_pred = 0
    v_fp_pred = 0

    # Classify ground truth islands
    if n_gt_sites > 0:
        for i in range(1, n_gt_sites + 1):
            gt_island_mask = gt_labeled == i
            island_volume = np.sum(gt_island_mask)
            if np.any(gt_island_mask & pred_binary_mask):
                v_found_gt += island_volume
            else:
                v_missed_gt += island_volume

    # Classify prediction islands
    if n_pred_islands > 0:
        for i in range(1, n_pred_islands + 1):
            pred_island_mask = pred_labeled == i
            island_volume = np.sum(pred_island_mask)
            if np.any(pred_island_mask & gt_binary_mask):
                v_tp_pred += island_volume
            else:
                if island_volume >= MIN_FP_VOXEL_COUNT:
                    v_fp_pred += island_volume

    v_gt_total = v_found_gt + v_missed_gt
    v_pred_total = v_tp_pred + v_fp_pred

    metrics = {
        "gt_found_volume": v_found_gt,
        "gt_missed_volume_fn": v_missed_gt,
        "pred_tp_volume": v_tp_pred,
        "pred_fp_volume_fp": v_fp_pred,
        "gt_total_volume": v_gt_total,
        "pred_total_volume": v_pred_total,
        "volume_recall": (
            v_found_gt / v_gt_total
            if v_gt_total > 0
            else (1.0 if v_pred_total == 0 else 0.0)
        ),
        "volume_precision": (
            v_tp_pred / v_pred_total
            if v_pred_total > 0
            else (1.0 if v_gt_total == 0 else 0.0)
        ),
    }
    return metrics


def main():
    console.rule(f"[bold blue]Evaluating Model with Island Volume Metrics[/bold blue]")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. Load Model ---
    console.print(f"[1/4] Loading trained model...")
    model = Deeper3DUnetWithDropout(in_channels=NUM_INPUT_CHANNELS, out_channels=1)

    if DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")
        model.to(DEVICE, memory_format=torch.channels_last_3d)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model = torch.compile(model)
    else:
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    console.print(f"  > Model loaded on [bold]{DEVICE}[/bold].")

    # --- 2. Load Test Set ---
    console.print(f"[2/4] Loading test set splits...")
    split_report_df = pd.read_csv(DATASET_SPLIT_REPORT_CSV)
    test_df = split_report_df[split_report_df["split"] == "test"].copy()
    all_test_pdbs = [(row.pdb_id, row.group) for row in test_df.itertuples()]

    # --- 3. Run Evaluation Loop ---
    console.print(
        f"[3/4] Running inference and calculating metrics for {len(all_test_pdbs)} proteins..."
    )
    all_metrics = []
    for pdb_id, protein_type in track(all_test_pdbs, description="Evaluating proteins"):
        h5_path = HDF5_CACHE_ROOT / f"{protein_type}_cache" / f"{pdb_id}.h5"
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            features_np = f["features"][:]
            target_mask_np_original = f["target_mask"][:]

        input_tensor = torch.from_numpy(features_np).to(torch.float32).to(DEVICE)
        prediction_np_padded, (pad_D, pad_H, pad_W) = run_sliding_window_inference(
            model, input_tensor
        )

        padded_target_mask_np = np.pad(
            target_mask_np_original, ((0, pad_D), (0, pad_H), (0, pad_W)), "constant"
        )

        protein_metrics = calculate_island_volume_metrics(
            prediction_np_padded, padded_target_mask_np
        )
        protein_metrics["pdb_id"] = pdb_id
        protein_metrics["type"] = protein_type
        all_metrics.append(protein_metrics)

    # --- 4. Aggregate and Display Results ---
    console.print("[4/4] Aggregating and saving results...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False, float_format="%.4f")
    console.print(f"  > Detailed metrics saved to [cyan]{METRICS_CSV_PATH}[/cyan]")

    pos_metrics = metrics_df[metrics_df["type"] == "positive"]

    # Focus on GT-side metrics
    table = Table(title=f"\nTest Set Evaluation Summary (N_pos={len(pos_metrics)})")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Positive Proteins (Mean ± Std)", justify="right", style="magenta")

    recall_mean = pos_metrics["volume_recall"].mean()
    recall_std = pos_metrics["volume_recall"].std()

    precision_mean = pos_metrics["volume_precision"].mean()
    precision_std = pos_metrics["volume_precision"].std()

    f1 = (
        2 * (precision_mean * recall_mean) / (precision_mean + recall_mean)
        if (precision_mean + recall_mean) > 0
        else 0
    )

    table.add_row(
        "Volume Recall (Found GT / Total GT)", f"{recall_mean:.3f} ± {recall_std:.3f}"
    )
    table.add_row(
        "Volume Precision (TP Pred / Total Pred)",
        f"{precision_mean:.3f} ± {precision_std:.3f}",
    )
    table.add_row("F1-Score (based on mean Recall/Precision)", f"{f1:.3f}")

    # Also report on total volumes to give context
    total_found_vol = pos_metrics["gt_found_volume"].sum()
    total_missed_vol = pos_metrics["gt_missed_volume_fn"].sum()
    total_fp_vol = pos_metrics["pred_fp_volume_fp"].sum()

    console.print(table)

    console.print("\n[bold]Overall Volume Distribution:[/bold]")
    console.print(
        f"  - Total Found GT Volume: [green]{total_found_vol:,.0f}[/green] voxels"
    )
    console.print(
        f"  - Total Missed GT Volume (FN): [yellow]{total_missed_vol:,.0f}[/yellow] voxels"
    )
    console.print(
        f"  - Total False Positive Volume (FP): [red]{total_fp_vol:,.0f}[/red] voxels"
    )

    console.rule("[bold green]Evaluation Complete[/bold green]")


if __name__ == "__main__":
    main()
