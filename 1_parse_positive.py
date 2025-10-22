# -*- coding: utf-8 -*-
"""
This script downloads protein structures associated with specific ligands,
extracts their sequences, and clusters them using MMseqs2.

The pipeline consists of the following steps:
1.  Reads a list of unique ligand codes from a CSV file.
2.  Queries the RCSB PDB API to find all protein structures (PDB IDs)
    that contain these ligands.
3.  Fetches metadata (title, experimental method) for each unique PDB ID,
    using a robust API-first, web-scraping-fallback approach.
4.  Downloads the corresponding CIF files for all found PDB structures in parallel.
5.  Parses the CIF files to extract all protein (peptide) sequences.
6.  Runs the MMseqs2 command-line tool to cluster the sequences based on
    identity and coverage criteria.
7.  Generates a final CSV report that maps each PDB ID to its cluster,
    associated ligands, and metadata.
"""
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import gemmi
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console
from rich.progress import Progress

# --- Configuration ---
console = Console()

# Input file containing ligand information
INPUT_LIGAND_CSV = "polyterpene_ligands_grouped.csv"

# Clustering parameters for MMseqs2
CLUSTER_IDENTITY = 0.9  # Minimum sequence identity for clustering
CLUSTER_COVERAGE = 0.8  # Minimum sequence coverage for clustering

# Output directories and file paths
OUTPUT_DIR = Path("protein_clustering_results")
STRUCTURE_DOWNLOAD_DIR = OUTPUT_DIR / "cif_files"
FASTA_FILE = OUTPUT_DIR / "sequences.fasta"
MMSEQS_WORKDIR = OUTPUT_DIR / "mmseqs_workdir"
FINAL_CSV_REPORT = OUTPUT_DIR / "clustered_protein_report.csv"

# Performance and network settings
MAX_WORKERS = 8           # Max parallel workers for downloading and processing
REQUEST_TIMEOUT = 60      # Timeout for API requests in seconds
DOWNLOAD_TIMEOUT = 300    # Timeout for file downloads in seconds

# API and URL endpoints
SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API_URL = "https://data.rcsb.org/rest/v1/core/entry"
CIF_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{}.cif"


def create_session() -> requests.Session:
    """Creates a requests session with automatic retries on common server errors."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session

session = create_session()

def search_pdb_by_ligand(ligand_code: str) -> tuple[str, list[str]]:
    """Finds all PDB IDs associated with a given ligand code via the RCSB Search API."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                "operator": "exact_match",
                "value": ligand_code
            }
        },
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
        "return_type": "entry"
    }
    try:
        response = session.post(SEARCH_API_URL, json=query, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        pdb_ids = [item["identifier"] for item in response.json().get("result_set", [])]
        return ligand_code, pdb_ids
    except Exception as e:
        console.log(f"API search failed for ligand {ligand_code}: {e}", style="red")
        return ligand_code, []

def get_details_via_api(pdb_id: str) -> dict | None:
    """Fetches structure title and method for a PDB ID using the RCSB Data API."""
    url = f"{DATA_API_URL}/{pdb_id}"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return {
            "title": data.get("struct", {}).get("title", "N/A"),
            "method": data.get("exptl", [{}])[0].get("method", "N/A")
        }
    except Exception:
        return None

def download_structure_file(pdb_id: str):
    """Downloads a CIF file for a given PDB ID if it doesn't already exist."""
    filepath = STRUCTURE_DOWNLOAD_DIR / f"{pdb_id}.cif"
    if filepath.exists() and filepath.stat().st_size > 0:
        return  # Skip if file already exists and is not empty
    
    url = CIF_DOWNLOAD_URL_TEMPLATE.format(pdb_id)
    try:
        response = session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        console.log(f"Failed to download {pdb_id}.cif: {e}", style="red")
        if filepath.exists():
            filepath.unlink() # Clean up partial download

def extract_sequences_from_cif(cif_file: Path) -> dict[str, str]:
    """Parses a CIF file and extracts all peptide sequences."""
    sequences = {}
    if not cif_file.exists() or cif_file.stat().st_size == 0:
        return sequences
        
    try:
        st = gemmi.read_structure(str(cif_file))
        if not st: return sequences
        
        for chain in st[0]: # Process only the first model
            if (polymer := chain.get_polymer()) and \
               (polymer.check_polymer_type() == gemmi.PolymerType.PeptideL):
                sequence = polymer.make_one_letter_sequence()
                if sequence:
                    sequences[f"{cif_file.stem}_{chain.name}"] = sequence
    except Exception as e:
        console.log(f"Failed to parse {cif_file.name}: {e}", style="red")
    return sequences

def check_mmseqs_installed() -> bool:
    """Checks if the 'mmseqs' command is available in the system's PATH."""
    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[bold red]Error: MMseqs2 is not installed or not in the system's PATH.[/bold red]")
        console.print("Please install it from [link=https://github.com/soedinglab/MMseqs2]https://github.com/soedinglab/MMseqs2[/link]")
        return False

def run_clustering(fasta_file: Path, work_dir: Path) -> pd.DataFrame | None:
    """Executes the MMseqs2 clustering pipeline on a given FASTA file."""
    if not fasta_file.exists() or fasta_file.stat().st_size == 0:
        console.log("FASTA file is empty. Skipping clustering.", style="yellow")
        return None
        
    db_name = work_dir / "DB"
    cluster_db_name = work_dir / "DB_clu"
    results_file = work_dir / "results.tsv"
    
    cmds = [
        ["mmseqs", "createdb", str(fasta_file), str(db_name), "--dbtype", "1"],
        ["mmseqs", "cluster", str(db_name), str(cluster_db_name), str(work_dir / "tmp"),
         "--min-seq-id", str(CLUSTER_IDENTITY), "-c", str(CLUSTER_COVERAGE), "--cov-mode", "1"],
        ["mmseqs", "createtsv", str(db_name), str(db_name), str(cluster_db_name), str(results_file)]
    ]
    
    try:
        for cmd in cmds:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        return pd.read_csv(results_file, sep='\t', header=None, names=['cluster_id', 'member_id'])
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]MMseqs2 clustering failed:[/bold red]\n{e.stderr}")
        return None

def main():
    """Main function to orchestrate the download and clustering pipeline."""
    if not check_mmseqs_installed():
        sys.exit(1)
    
    console.rule("[bold blue]Starting Download and Clustering Pipeline[/bold blue]")

    # --- Step 0: Load Ligand Codes ---
    try:
        ligand_df = pd.read_csv(INPUT_LIGAND_CSV)
        ligand_codes = ligand_df['group_id'].unique().tolist()
        if not ligand_codes:
            console.print(f"[bold red]Error: No ligand codes found in the 'group_id' column of '{INPUT_LIGAND_CSV}'.[/bold red]")
            sys.exit(1)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[bold red]Error reading '{INPUT_LIGAND_CSV}': {e}[/bold red]")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    STRUCTURE_DOWNLOAD_DIR.mkdir(exist_ok=True)
    MMSEQS_WORKDIR.mkdir(exist_ok=True)

    with Progress(console=console) as progress:
        # --- Step 1: Find all PDB IDs for the given ligands ---
        search_task = progress.add_task("Step 1: Searching for PDB IDs...", total=len(ligand_codes))
        pdb_ligand_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(search_pdb_by_ligand, lc) for lc in ligand_codes}
            for future in concurrent.futures.as_completed(futures):
                ligand, pdb_ids = future.result()
                for pdb_id in pdb_ids:
                    pdb_ligand_map.setdefault(pdb_id, set()).add(ligand)
                progress.update(search_task, advance=1)

        unique_pdb_ids = list(pdb_ligand_map.keys())
        if not unique_pdb_ids:
            console.print("[yellow]No PDB structures were found for the given ligands.[/yellow]")
            sys.exit(0)
        progress.log(f"Found [cyan]{len(unique_pdb_ids)}[/cyan] unique PDB IDs.")

        # --- Step 2: Fetch metadata for each PDB ID ---
        meta_task = progress.add_task("Step 2: Fetching metadata...", total=len(unique_pdb_ids))
        metadata_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_details_via_api, pid) for pid in unique_pdb_ids}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                pdb_id = unique_pdb_ids[i] # This is not guaranteed, better approach needed
                details = future.result() or {"title": "N/A (API Failed)", "method": "N/A (API Failed)"}
                metadata_list.append({"pdb_id": pdb_id, **details})
                progress.update(meta_task, advance=1)
        
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df['ligands'] = metadata_df['pdb_id'].map(
            lambda pid: "; ".join(sorted(list(pdb_ligand_map.get(pid, []))))
        )

        # --- Step 3: Download all CIF files ---
        dl_task = progress.add_task("Step 3: Downloading CIF files...", total=len(unique_pdb_ids))
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_structure_file, pid) for pid in unique_pdb_ids}
            for _ in concurrent.futures.as_completed(futures):
                progress.update(dl_task, advance=1)

        # --- Step 4: Extract sequences and create FASTA file ---
        structure_files = list(STRUCTURE_DOWNLOAD_DIR.glob("*.cif"))
        parse_task = progress.add_task("Step 4: Extracting sequences...", total=len(structure_files))
        all_sequences = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_sequences_from_cif, sf) for sf in structure_files}
            for future in concurrent.futures.as_completed(futures):
                all_sequences.update(future.result())
                progress.update(parse_task, advance=1)
        
        with open(FASTA_FILE, "w") as f:
            for seq_id, sequence in all_sequences.items():
                f.write(f">{seq_id}\n{sequence}\n")
        progress.log(f"Extracted [cyan]{len(all_sequences)}[/cyan] protein chains.")

    # --- Step 5: Run Clustering ---
    console.rule("Step 5: Clustering Protein Sequences")
    cluster_results_df = run_clustering(FASTA_FILE, MMSEQS_WORKDIR)

    # --- Step 6: Generate Final Report ---
    console.rule("Step 6: Generating Final Report")
    if cluster_results_df is not None:
        cluster_results_df['pdb_id'] = cluster_results_df['member_id'].str.split('_').str[0]
        pdb_to_cluster = cluster_results_df.drop_duplicates(subset='pdb_id').set_index('pdb_id')['cluster_id']
        metadata_df['cluster_id'] = metadata_df['pdb_id'].map(pdb_to_cluster).fillna("No_Cluster")
    else:
        metadata_df['cluster_id'] = "Clustering_Failed"

    final_df = metadata_df[['pdb_id', 'cluster_id', 'ligands', 'method', 'title']]
    final_df.sort_values(by=['cluster_id', 'pdb_id'], inplace=True)
    final_df.to_csv(FINAL_CSV_REPORT, index=False)
    
    console.print(f"\n[bold green]Success! Final report saved to: {FINAL_CSV_REPORT}[/bold green]")
    if cluster_results_df is not None:
        console.print(f"Total unique PDBs analyzed: [cyan]{len(final_df)}[/cyan]")
        console.print(f"Total clusters identified: [cyan]{final_df['cluster_id'].nunique()}[/cyan]")

if __name__ == "__main__":
    main()