# -*- coding: utf-8 -*-
"""
Evaluates the trained 3D U-Net model on the held-out test set.

This script provides a final, unbiased assessment of the model's performance
by running inference on the test split, which was not used during training or
validation. It calculates a comprehensive set of segmentation metrics.

The pipeline consists of the following steps:
1.  Loads the test set definitions from the 'dataset_splits.json' file.
2.  Loads the pre-trained model weights.
3.  Iterates through each protein in the test set (both positive and negative).
4.  For each protein:
    a. Loads the pre-calculated feature and ground truth tensors from the HDF5 cache.
    b. Runs inference using the sliding window approach.
    c. Calculates a suite of evaluation metrics by comparing the model's
       prediction map against the ground truth mask.
5.  Aggregates the results and prints a summary table of mean metrics and their
    standard deviations to the console.
6.  Saves a detailed CSV file containing the metrics for every individual
    protein in the test set for fine-grained analysis and anomaly detection.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    auc,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# --- Import components from training and prediction scripts ---
from inference import run_sliding_window_inference
from model import Deeper3DUnetWithDropout

# Suppress benign warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Configuration ---
console = Console()

# --- INPUT: Paths to data, splits, and trained model ---
HDF5_CACHE_ROOT = Path("feature_cache")
DATASET_SPLITS_JSON = Path("patch_coordinates_cache/dataset_splits.json")
MODEL_SAVE_PATH = Path("unet_training_output/unet_best_model.pth")
REPORT_POSITIVE_CSV = Path("protein_clustering_results/clustered_protein_report.csv")
REPORT_NEGATIVE_CSV = Path(
    "negative_controls_results/negative_controls_results_report.csv"
)

# --- OUTPUT ---
OUTPUT_DIR = Path("evaluation_results")
METRICS_CSV_PATH = OUTPUT_DIR / "test_set_metrics_per_protein.csv"

# --- Model and Inference Parameters (Must match) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INPUT_CHANNELS = 8


def calculate_metrics(
    prediction_map: np.ndarray, ground_truth_mask: np.ndarray
) -> Dict:
    """Calculates a comprehensive set of metrics for a single protein."""
    metrics = {}

    # Flatten arrays for sklearn metrics
    pred_flat = prediction_map.flatten()
    true_flat = ground_truth_mask.flatten().astype(np.uint8)

    # Handle cases with no positive class in ground truth (negative controls)
    if true_flat.sum() == 0:
        metrics["roc_auc"] = np.nan
        metrics["auprc"] = np.nan
        metrics["dice_f1"] = 1.0 if pred_flat.max() < 0.5 else 0.0
        metrics["iou_jaccard"] = 1.0 if pred_flat.max() < 0.5 else 0.0
        metrics["precision"] = 1.0 if pred_flat.max() < 0.5 else 0.0
        metrics["recall"] = 1.0
        metrics["mean_prediction_value"] = pred_flat.mean()
        return metrics

    # Threshold-independent metrics
    metrics["roc_auc"] = roc_auc_score(true_flat, pred_flat)
    precision_curve, recall_curve, _ = precision_recall_curve(true_flat, pred_flat)
    metrics["auprc"] = auc(recall_curve, precision_curve)

    # Threshold-based metrics (at 0.5)
    pred_binary_flat = (pred_flat > 0.5).astype(np.uint8)

    metrics["dice_f1"] = f1_score(true_flat, pred_binary_flat, zero_division=0)
    metrics["iou_jaccard"] = jaccard_score(true_flat, pred_binary_flat, zero_division=0)
    metrics["precision"] = precision_score(true_flat, pred_binary_flat, zero_division=0)
    metrics["recall"] = recall_score(true_flat, pred_binary_flat, zero_division=0)
    metrics["mean_prediction_value"] = pred_flat.mean()

    return metrics


def main():
    """Main function to orchestrate the evaluation pipeline."""
    console.rule(f"[bold blue]Evaluating Model on Test Set[/bold blue]")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. Load Model ---
    console.print(f"[1/4] Loading trained model from [cyan]{MODEL_SAVE_PATH}[/cyan]...")
    if not MODEL_SAVE_PATH.exists():
        console.print(f"[bold red]Error: Trained model not found.[/bold red]")
        return

    model = Deeper3DUnetWithDropout(in_channels=NUM_INPUT_CHANNELS, out_channels=1)
    if DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")
        model.to(DEVICE, memory_format=torch.channels_last_3d)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model = torch.compile(model)
    else:
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    console.print(f"  > Model loaded and compiled on [bold]{DEVICE}[/bold].")

    # --- 2. Load Test Set PDB IDs ---
    console.print(
        f"[2/4] Loading test set splits from [cyan]{DATASET_SPLITS_JSON}[/cyan]..."
    )
    with open(DATASET_SPLITS_JSON, "r") as f:
        splits = json.load(f)

    df_pos = pd.read_csv(REPORT_POSITIVE_CSV)
    df_neg = pd.read_csv(REPORT_NEGATIVE_CSV)
    cluster_to_pdb = (
        pd.concat([df_pos, df_neg])
        .groupby("cluster_id")["pdb_id"]
        .apply(list)
        .to_dict()
    )

    test_pdb_ids_pos = [
        p for c in splits["test"]["positive"] for p in cluster_to_pdb.get(c, [])
    ]
    test_pdb_ids_neg = [
        p for c in splits["test"]["negative"] for p in cluster_to_pdb.get(c, [])
    ]

    all_test_pdbs = [(pdb_id, "positive") for pdb_id in test_pdb_ids_pos] + [
        (pdb_id, "negative") for pdb_id in test_pdb_ids_neg
    ]

    console.print(
        f"  > Found {len(test_pdb_ids_pos)} positive and {len(test_pdb_ids_neg)} negative proteins in the test set."
    )

    # --- 3. Run Inference and Calculate Metrics ---
    console.print(
        f"[3/4] Running inference and calculating metrics for {len(all_test_pdbs)} proteins..."
    )
    all_metrics = []

    for pdb_id, protein_type in tqdm(all_test_pdbs, desc="Evaluating proteins"):
        h5_dir = HDF5_CACHE_ROOT / f"{protein_type}_cache"
        h5_path = h5_dir / f"{pdb_id}.h5"

        if not h5_path.exists():
            console.log(
                f"[yellow]Warning: HDF5 file not found for {pdb_id}, skipping.[/yellow]"
            )
            continue

        with h5py.File(h5_path, "r") as f:
            features_np = f["features"][:]
            target_mask_np = f["target_mask"][:]

        input_tensor = torch.from_numpy(features_np).to(torch.float32).to(DEVICE)

        # Run inference
        prediction_np = run_sliding_window_inference(model, input_tensor)

        # Calculate metrics
        protein_metrics = calculate_metrics(prediction_np, target_mask_np)
        protein_metrics["pdb_id"] = pdb_id
        protein_metrics["type"] = protein_type
        all_metrics.append(protein_metrics)

    # --- 4. Save and Display Results ---
    console.print("[4/4] Aggregating and saving results...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False, float_format="%.4f")
    console.print(
        f"  > Detailed per-protein metrics saved to [cyan]{METRICS_CSV_PATH}[/cyan]"
    )

    # Display summary table
    pos_metrics = metrics_df[metrics_df["type"] == "positive"]
    neg_metrics = metrics_df[metrics_df["type"] == "negative"]

    table = Table(
        title=f"\nTest Set Evaluation Summary (N_pos={len(pos_metrics)}, N_neg={len(neg_metrics)})"
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Positive Proteins (Mean ± Std)", justify="right", style="magenta")
    table.add_column("Negative Proteins (Mean ± Std)", justify="right", style="green")

    metric_keys_pos = [
        "roc_auc",
        "auprc",
        "dice_f1",
        "iou_jaccard",
        "precision",
        "recall",
    ]
    for key in metric_keys_pos:
        mean = pos_metrics[key].mean()
        std = pos_metrics[key].std()
        table.add_row(key.upper(), f"{mean:.3f} ± {std:.3f}", "---")

    mean_pred_pos = pos_metrics["mean_prediction_value"].mean()
    std_pred_pos = pos_metrics["mean_prediction_value"].std()
    mean_pred_neg = neg_metrics["mean_prediction_value"].mean()
    std_pred_neg = neg_metrics["mean_prediction_value"].std()

    table.add_row(
        "Mean Prediction Value",
        f"{mean_pred_pos:.4f} ± {std_pred_pos:.4f}",
        f"{mean_pred_neg:.4f} ± {std_pred_neg:.4f}",
    )

    console.print(table)
    console.rule("[bold green]Evaluation Complete[/bold green]")


if __name__ == "__main__":
    main()
