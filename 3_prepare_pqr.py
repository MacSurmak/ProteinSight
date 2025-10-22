# -*- coding: utf-8 -*-
"""
Prepare PQR files for all datasets.

This script automates the full preparation pipeline for protein structures
intended for docking or electrostatic analysis. It processes multiple datasets
(e.g., positive controls, negative controls, albumins) in a parallel and
robust manner.

Key Features:
-   Preserves intermediate .pdb files (converted from .cif) in separate,
    organized directories for each dataset, facilitating debugging and reuse.
-   Collects all processing errors and presents them in a consolidated
    summary report at the end, rather than printing them in real-time.
-   Leverages multiprocessing to significantly speed up the conversion
    of large numbers of files.

The pipeline for each structure is as follows:
1.  Automatically discovers all `.cif` files in the source directories.
2.  Performs a two-step conversion for each file:
    a) `.cif` -> `.pdb` (using gemmi, saved).
    b) `.pdb` -> `.pqr` (using pdb2pqr, saved).
"""
import concurrent.futures
import subprocess
from pathlib import Path

import gemmi
from rich.console import Console
from rich.progress import Progress

# --- Configuration ---
console = Console()

# Configuration defining input/output paths for each dataset.
# Each dataset will have its own dedicated directories for PDB and PQR files.
DATASET_PATHS = {
    "positive_controls": {
        "cif_dir": Path("protein_clustering_results/cif_files"),
        "pdb_dir": Path("positive_pdb_files"),
        "pqr_dir": Path("positive_pqr_files"),
    },
    "negative_controls": {
        "cif_dir": Path("negative_controls_results/cif_files"),
        "pdb_dir": Path("negative_pdb_files"),
        "pqr_dir": Path("negative_pqr_files"),
    },
    "albumins": {
        "cif_dir": Path("albumins_results/cif_files"),
        "pdb_dir": Path("albumins_pdb_files"),
        "pqr_dir": Path("albumins_pqr_files"),
    },
}

# Parameters for the pdb2pqr tool
FORCE_FIELD = "AMBER"
PH_VALUE = "7.4"

# Performance settings
MAX_WORKERS = 16


def cif_to_pdb(cif_path: Path, pdb_path: Path):
    """
    Converts a .cif file to a .pdb file using the gemmi library.

    Args:
        cif_path: Path to the source .cif file.
        pdb_path: Path to the destination .pdb file.

    Raises:
        RuntimeError: If gemmi fails to read or write the structure.
    """
    try:
        structure = gemmi.read_structure(str(cif_path))
        # Write PDB format without TER records for better compatibility
        structure.write_pdb(str(pdb_path), ter_records=False)
    except Exception as e:
        # This error will be caught and reported by the calling function.
        raise RuntimeError(f"gemmi error while converting {cif_path.name}") from e


def process_structure(task: tuple[Path, Path, Path]) -> tuple[str, bool, str]:
    """
    Executes the full processing pipeline for a single structure: .cif -> .pdb -> .pqr.

    This function is designed to be run in a separate process.

    Args:
        task: A tuple containing (cif_path, pdb_dir, pqr_dir).

    Returns:
        A tuple containing: (pdb_id, success_flag, error_message).
        The error message is an empty string on success.
    """
    cif_path, pdb_dir, pqr_dir = task
    pdb_id = cif_path.stem
    
    final_pdb_path = pdb_dir / f"{pdb_id}.pdb"
    final_pqr_path = pqr_dir / f"{pdb_id}.pqr"

    try:
        # Step 1: Convert .cif to .pdb
        cif_to_pdb(cif_path, final_pdb_path)

        # Step 2: Run pdb2pqr to convert .pdb to .pqr
        command = [
            "pdb2pqr",
            f"--ff={FORCE_FIELD}",
            f"--with-ph={PH_VALUE}",
            "--drop-water",
            str(final_pdb_path),
            str(final_pqr_path),
        ]

        # Execute the command, capturing output and checking for errors.
        subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        
        # On success, return a success status with no error message.
        return pdb_id, True, ""
        
    except FileNotFoundError:
        return pdb_id, False, "'pdb2pqr' command not found. Ensure it is installed and in the system's PATH."
    except subprocess.CalledProcessError as e:
        # If pdb2pqr fails, return its stderr for diagnostics.
        return pdb_id, False, e.stderr.strip()
    except Exception as e:
        # Catch any other unexpected errors, e.g., from cif_to_pdb.
        return pdb_id, False, f"An unexpected error occurred: {str(e)}"


def main():
    """Main function to orchestrate the PQR file preparation process."""
    console.rule("[bold blue]STAGE 1: Preparing PQR and PDB files[/bold blue]")

    tasks_to_process = []
    # Discover all .cif files and prepare the list of tasks.
    for name, paths in DATASET_PATHS.items():
        cif_dir = paths["cif_dir"]
        pdb_dir = paths["pdb_dir"]
        pqr_dir = paths["pqr_dir"]
        
        pdb_dir.mkdir(exist_ok=True)
        pqr_dir.mkdir(exist_ok=True)
        
        if not cif_dir.exists():
            console.print(f"[yellow]Warning: Directory {cif_dir} not found. Skipping dataset '[bold]{name}[/bold]'.[/yellow]")
            continue
            
        cif_files = list(cif_dir.glob("*.cif"))
        console.print(f"Found {len(cif_files)} .cif files for the '[bold cyan]{name}[/bold cyan]' dataset.")
        for cif_file in cif_files:
            tasks_to_process.append((cif_file, pdb_dir, pqr_dir))

    if not tasks_to_process:
        console.print("[bold red]No .cif files found to process.[/bold red]")
        return
        
    console.print(f"\nTotal tasks to process: {len(tasks_to_process)}")
    console.print(f"Running with {MAX_WORKERS} parallel processes...")

    successful_count = 0
    failed_tasks = []

    # Process all tasks in parallel with a progress bar.
    with Progress(console=console) as progress:
        task_id = progress.add_task("Preparing files...", total=len(tasks_to_process))
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all jobs and create a map of futures to tasks.
            future_to_task = {executor.submit(process_structure, task): task for task in tasks_to_process}
            
            for future in concurrent.futures.as_completed(future_to_task):
                pdb_id, success, msg = future.result()
                if success:
                    successful_count += 1
                else:
                    failed_tasks.append((pdb_id, msg))
                progress.update(task_id, advance=1)

    # --- Final Report ---
    console.rule("[bold green]Preparation Complete[/bold green]")
    console.print(f"Successfully processed: [green]{successful_count}[/green] files.")
    
    if failed_tasks:
        failed_count = len(failed_tasks)
        console.print(f"Failed to process: [red]{failed_count}[/red] files.")
        console.print("\n[bold red]Error Summary:[/bold red]")
        for pdb_id, error_msg in sorted(failed_tasks):
            # Print the last, most informative line of the error message.
            last_error_line = error_msg.splitlines()[-1] if error_msg.splitlines() else "Unknown error"
            console.print(f" - [bold cyan]{pdb_id}[/bold cyan]: {last_error_line}")
    else:
        console.print("All files were processed without errors.")

    console.print("\nPrepared files are located in the following directories:")
    for name, paths in DATASET_PATHS.items():
        if paths["cif_dir"].exists():
             console.print(f"\n[bold]{name.replace('_', ' ').title()}:[/bold]")
             console.print(f" - PDB files: [cyan]{paths['pdb_dir']}[/cyan]")
             console.print(f" - PQR files: [cyan]{paths['pqr_dir']}[/cyan]")


if __name__ == "__main__":
    main()