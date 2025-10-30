# -*- coding: utf-8 -*-
"""
Evaluates the trained 3D U-Net model on the held-out test set.

This script provides a final, unbiased assessment of the model's performance
by running inference on the test split, which was not used during training or
validation. It calculates a comprehensive set of segmentation metrics.

The pipeline consists of the following steps:
1.  Loads the test set definitions from the 'dataset_split_report.csv' file.
2.  Loads the pre-trained model weights.
3.  Iterates through each protein in the test set (both positive and negative).
4.  For each protein:
    a. Loads the pre-calculated feature and ground truth tensors from the HDF5 cache.
    b. Ensures consistent tensor dimensions by padding the ground truth mask to
       match the padded prediction output from the sliding window inference.
    c. Calculates a suite of evaluation metrics by comparing the model's
       prediction map against the ground truth mask.
5.  Aggregates the results and prints a summary table of mean metrics and their
    standard deviations to the console.
6.  Saves a detailed CSV file containing the metrics for every individual
    protein in the test set for fine-grained analysis.
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
from sklearn.metrics import (
    auc,
    f1_score,
    jaccard_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from inference import run_sliding_window_inference
from model import Deeper3DUnetWithDropout

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# --- Configuration ---
console = Console()

HDF5_CACHE_ROOT = Path("feature_cache")
DATASET_SPLIT_REPORT_CSV = Path("dataset_split_report.csv")
MODEL_SAVE_PATH = Path("unet_training_output/unet_best_model.pth")

OUTPUT_DIR = Path("evaluation_results")
METRICS_CSV_PATH = OUTPUT_DIR / "test_set_metrics_per_protein.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INPUT_CHANNELS = 8
METRIC_THRESHOLD = 0.01


def calculate_metrics(
    prediction_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = METRIC_THRESHOLD,
) -> Dict[str, float]:
    """
    Calculates segmentation metrics after binarizing both inputs by a threshold.
    """
    metrics = {}
    pred_flat = prediction_map.flatten()
    true_flat_continuous = ground_truth_mask.flatten()

    pred_binary_flat = (pred_flat > threshold).astype(np.uint8)
    true_binary_flat = (true_flat_continuous > threshold).astype(np.uint8)

    # Handle cases with no positive class in the ground truth (e.g., negative samples)
    if true_binary_flat.sum() == 0:
        metrics["roc_auc"] = np.nan
        metrics["auprc"] = np.nan
        is_correct = pred_binary_flat.sum() == 0
        metrics["dice_f1"] = 1.0 if is_correct else 0.0
        metrics["iou_jaccard"] = 1.0 if is_correct else 0.0
        metrics["precision"] = 1.0 if is_correct else 0.0
        metrics["recall"] = 1.0 if is_correct else 0.0
        metrics["mean_prediction_value"] = pred_flat.mean()
        return metrics

    # Calculate metrics for samples with a positive class
    try:
        metrics["roc_auc"] = roc_auc_score(true_binary_flat, pred_flat)
        precision_curve, recall_curve, _ = precision_recall_curve(
            true_binary_flat, pred_flat
        )
        metrics["auprc"] = auc(recall_curve, precision_curve)
    except ValueError:
        metrics["roc_auc"] = np.nan
        metrics["auprc"] = np.nan

    metrics["dice_f1"] = f1_score(true_binary_flat, pred_binary_flat, zero_division=0)
    metrics["iou_jaccard"] = jaccard_score(
        true_binary_flat, pred_binary_flat, zero_division=0
    )
    metrics["precision"] = precision_score(
        true_binary_flat, pred_binary_flat, zero_division=0
    )
    metrics["recall"] = recall_score(
        true_binary_flat, pred_binary_flat, zero_division=0
    )
    metrics["mean_prediction_value"] = pred_flat.mean()

    return metrics


def main():
    console.rule(f"[bold blue]Evaluating Model on Test Set[/bold blue]")
    OUTPUT_DIR.mkdir(exist_ok=True)

    console.print(f"[1/4] Loading trained model from [cyan]{MODEL_SAVE_PATH}[/cyan]...")
    if not MODEL_SAVE_PATH.exists():
        console.print(f"[bold red]Error: Trained model not found.[/bold red]")
        return

    model = Deeper3DUnetWithDropout(in_channels=NUM_INPUT_CHANNELS, out_channels=1)
    if DEVICE == "cuda":
        # torch.set_float32_matmul_precision("high")
        model.to(DEVICE, memory_format=torch.channels_last_3d)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model = torch.compile(model)
    else:
        model.to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    console.print(f"  > Model loaded and compiled on [bold]{DEVICE}[/bold].")

    console.print(
        f"[2/4] Loading test set splits from [cyan]{DATASET_SPLIT_REPORT_CSV}[/cyan]..."
    )
    split_report_df = pd.read_csv(DATASET_SPLIT_REPORT_CSV)
    test_df = split_report_df[split_report_df["split"] == "test"].copy()

    test_pdb_ids_pos = test_df[test_df["group"] == "positive"]["pdb_id"].tolist()
    test_pdb_ids_neg = test_df[test_df["group"] == "negative"]["pdb_id"].tolist()

    all_test_pdbs = [(pdb_id, "positive") for pdb_id in test_pdb_ids_pos] + [
        (pdb_id, "negative") for pdb_id in test_pdb_ids_neg
    ]
    console.print(
        f"  > Found {len(test_pdb_ids_pos)} positive and {len(test_pdb_ids_neg)} negative proteins."
    )

    console.print(
        f"[3/4] Running inference and calculating metrics for {len(all_test_pdbs)} proteins..."
    )
    all_metrics = []

    for pdb_id, protein_type in track(all_test_pdbs, description="Evaluating proteins"):
        h5_path = HDF5_CACHE_ROOT / f"{protein_type}_cache" / f"{pdb_id}.h5"
        if not h5_path.exists():
            console.log(
                f"[yellow]Warning: HDF5 file not found for {pdb_id}, skipping.[/yellow]"
            )
            continue

        with h5py.File(h5_path, "r") as f:
            features_np = f["features"][:]
            target_mask_np_original = f["target_mask"][:]

        input_tensor = torch.from_numpy(features_np).to(torch.float32).to(DEVICE)
        prediction_np_padded, (pad_D, pad_H, pad_W) = run_sliding_window_inference(
            model, input_tensor
        )

        padded_target_mask_np = np.pad(
            target_mask_np_original,
            ((0, pad_D), (0, pad_H), (0, pad_W)),
            mode="constant",
            constant_values=0,
        )
        assert prediction_np_padded.shape == padded_target_mask_np.shape

        protein_metrics = calculate_metrics(prediction_np_padded, padded_target_mask_np)
        protein_metrics["pdb_id"] = pdb_id
        protein_metrics["type"] = protein_type
        all_metrics.append(protein_metrics)

    console.print("[4/4] Aggregating and saving results...")
    if not all_metrics:
        console.print(
            "[bold red]Error: No metrics calculated. Check HDF5 files.[/bold red]"
        )
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False, float_format="%.4f")
    console.print(
        f"  > Detailed per-protein metrics saved to [cyan]{METRICS_CSV_PATH}[/cyan]"
    )

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
