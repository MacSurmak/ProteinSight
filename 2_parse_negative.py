# -*- coding: utf-8 -*-
"""
This script downloads and processes two distinct sets of "negative control"
protein structures from the RCSB PDB for comparative analysis.

The script performs two main tasks:
1.  General Negative Controls:
    -   It queries the PDB for proteins associated with common, non-terpene
      ligands (e.g., nucleotides, steroids, fatty acids, hemes).
    -   To create a diverse yet manageable dataset, it takes a random sample
      of PDB IDs from the search results for each ligand group.
    -   It then processes this combined set by downloading CIF files, extracting
      protein sequences, clustering them with MMseqs2, and generating a final report.

2.  Albumin Controls:
    -   It performs a text-based search for "serum albumin" to gather a specific
      set of highly abundant, known small-molecule binders.
    -   It takes a random sample from these results and performs the same
      download, extraction, and clustering pipeline.

Each task results in a separate output directory with structures, sequences,
and a clustered report, allowing for independent analysis of these control groups.
"""
import concurrent.futures
import random
import subprocess
import sys
from pathlib import Path

import gemmi
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console
from rich.progress import Progress

# --- Configuration ---
console = Console()

# Define groups of common ligands to serve as negative controls.
# For each group, a random sample of structures will be taken.
NEGATIVE_CONTROL_TARGETS = {
    "nucleotides": {"codes": ["ATP", "GTP", "ADP", "ANP"], "sample_size": 300},
    "steroids": {"codes": ["CHO", "EST", "TES", "DXM"], "sample_size": 300},
    "fatty_acids": {"codes": ["PLM", "STE", "OLA"], "sample_size": 300},
    "hemes_porphyrins": {"codes": ["HEM", "HEA", "HEC"], "sample_size": 300},
}

# Define a specific text search for serum albumins.
ALBUMIN_TARGET = {"search_term": "serum albumin", "sample_size": 300}

# Clustering and performance settings
CLUSTER_IDENTITY = 0.9
CLUSTER_COVERAGE = 0.8
MAX_WORKERS = 8
REQUEST_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 300

# Directory and API settings
NEG_CONTROLS_BASE_DIR = Path("negative_controls_results")
ALBUMINS_BASE_DIR = Path("albumins_results")
SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API_URL = "https://data.rcsb.org/rest/v1/core/entry"
CIF_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{}.cif"

def create_session() -> requests.Session:
    """Creates a requests session with automatic retries for server errors."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session

session = create_session()

def search_pdb_by_ligand_codes(ligand_codes: list[str]) -> list[str]:
    """Finds PDB IDs associated with any of the provided ligand codes."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                        "operator": "exact_match",
                        "value": code
                    }
                } for code in ligand_codes
            ]
        },
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
        "return_type": "entry"
    }
    try:
        response = session.post(SEARCH_API_URL, json=query, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return [item["identifier"] for item in response.json().get("result_set", [])]
    except Exception as e:
        console.log(f"API search failed for ligands {ligand_codes}: {e}", style="red")
        return []

def search_pdb_by_text(search_term: str) -> list[str]:
    """Finds PDB IDs by performing a text search on the structure title."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "contains_phrase",
                "value": search_term,
                "attribute": "struct.title"
            }
        },
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
        "return_type": "entry"
    }
    try:
        response = session.post(SEARCH_API_URL, json=query, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return [item["identifier"] for item in response.json().get("result_set", [])]
    except Exception as e:
        console.log(f"API text search failed for '{search_term}': {e}", style="red")
        return []

def get_details_via_api(pdb_id: str) -> dict | None:
    """Fetches structure title and method for a PDB ID via the Data API."""
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

def fetch_and_parse_details(pdb_id: str) -> dict:
    """Fetches PDB entry details using the API as the primary source."""
    details = get_details_via_api(pdb_id)
    if details is None:
        details = {"title": "API Fetch Failed", "method": "API Fetch Failed"}
    return {"pdb_id": pdb_id, **details}

def download_structure_file(pdb_id: str, download_dir: Path):
    """Downloads a CIF file for a given PDB ID if it doesn't already exist."""
    filepath = download_dir / f"{pdb_id}.cif"
    if filepath.exists() and filepath.stat().st_size > 0:
        return
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
        for chain in st[0]:
            if (polymer := chain.get_polymer()) and \
               (polymer.check_polymer_type() == gemmi.PolymerType.PeptideL):
                sequence = polymer.make_one_letter_sequence()
                if sequence:
                    sequences[f"{cif_file.stem}_{chain.name}"] = sequence
    except Exception as e:
        console.log(f"Failed to parse {cif_file.name}: {e}", style="red")
    return sequences

def check_mmseqs_installed() -> bool:
    """Checks if the 'mmseqs' command is available."""
    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[bold red]Error: MMseqs2 is not installed or not in the system's PATH.[/bold red]")
        return False

def run_clustering(fasta_file: Path, work_dir: Path) -> pd.DataFrame | None:
    """Executes the MMseqs2 clustering pipeline."""
    if not fasta_file.exists() or fasta_file.stat().st_size == 0:
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

def process_dataset(title: str, records: list[dict], base_dir: Path):
    """Runs the full processing pipeline for a given dataset of PDB IDs."""
    console.rule(f"[bold blue]Processing Dataset: {title}[/bold blue]")
    if not records:
        console.print("[yellow]No records to process for this dataset.[/yellow]")
        return

    pdb_ids_to_process = sorted(list(set(r['pdb_id'] for r in records)))
    
    cif_dir = base_dir / "cif_files"
    mmseqs_dir = base_dir / "mmseqs_workdir"
    fasta_file = base_dir / "sequences.fasta"
    report_file = base_dir / f"{base_dir.name}_report.csv"
    
    for dir_path in [base_dir, cif_dir, mmseqs_dir]:
        dir_path.mkdir(exist_ok=True)
    
    with Progress(console=console) as progress:
        # Fetch metadata
        meta_task = progress.add_task("Fetching metadata...", total=len(pdb_ids_to_process))
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_and_parse_details, pid) for pid in pdb_ids_to_process}
            metadata = [future.result() for future in concurrent.futures.as_completed(futures)]
            progress.update(meta_task, completed=len(metadata))

        # Download CIF files
        dl_task = progress.add_task("Downloading CIF files...", total=len(pdb_ids_to_process))
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(download_structure_file, pid, cif_dir) for pid in pdb_ids_to_process}
            for _ in concurrent.futures.as_completed(futures):
                progress.update(dl_task, advance=1)
        
        # Extract sequences
        structure_files = list(cif_dir.glob("*.cif"))
        parse_task = progress.add_task("Extracting sequences...", total=len(structure_files))
        all_sequences = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_sequences_from_cif, sf) for sf in structure_files}
            for future in concurrent.futures.as_completed(futures):
                all_sequences.update(future.result())
                progress.update(parse_task, advance=1)

    with open(fasta_file, "w") as f:
        for seq_id, seq in all_sequences.items():
            f.write(f">{seq_id}\n{seq}\n")
    console.print(f"Extracted {len(all_sequences)} protein chains.")
    
    # Run clustering and generate report
    cluster_df = run_clustering(fasta_file, mmseqs_dir)
    
    report_df = pd.DataFrame(records)
    metadata_df = pd.DataFrame(metadata)
    report_df = pd.merge(report_df, metadata_df, on="pdb_id", how="left")
    
    if cluster_df is not None:
        cluster_df['pdb_id'] = cluster_df['member_id'].str.split('_').str[0]
        pdb_to_cluster = cluster_df.drop_duplicates(subset='pdb_id').set_index('pdb_id')['cluster_id']
        report_df['cluster_id'] = report_df['pdb_id'].map(pdb_to_cluster).fillna("No_Cluster")
    else:
        report_df['cluster_id'] = "Clustering_Failed"

    final_cols = ['pdb_id', 'cluster_id', 'group', 'method', 'title']
    report_df = report_df[final_cols]
    report_df.sort_values(by=['cluster_id', 'pdb_id'], inplace=True)
    report_df.to_csv(report_file, index=False)
    console.print(f"[bold green]Report for '{title}' saved to: {report_file}[/bold green]")
    console.print(f"Total structures: {len(report_df)}, Unique clusters: {report_df['cluster_id'].nunique()}")

def main():
    """Main function to orchestrate the collection of all negative control sets."""
    if not check_mmseqs_installed():
        sys.exit(1)

    # --- Task 1: Process General Negative Controls ---
    console.rule("[bold]Collecting General Negative Controls[/bold]")
    all_neg_control_records = []
    processed_pdb_ids = set()
    for group_name, data in NEGATIVE_CONTROL_TARGETS.items():
        console.print(f"\nSearching for '{group_name}' (ligands: {', '.join(data['codes'])})...")
        pdb_ids = search_pdb_by_ligand_codes(data['codes'])
        unique_new_ids = [pid for pid in pdb_ids if pid not in processed_pdb_ids]
        console.print(f"Found {len(unique_new_ids)} new unique structures.")
        
        sample_size = min(len(unique_new_ids), data['sample_size'])
        selected_ids = random.sample(unique_new_ids, sample_size)
        console.print(f"Randomly selected {len(selected_ids)} structures for this group.")
        
        for pid in selected_ids:
            all_neg_control_records.append({"pdb_id": pid, "group": group_name})
        processed_pdb_ids.update(selected_ids)
    
    process_dataset("Negative Controls", all_neg_control_records, NEG_CONTROLS_BASE_DIR)

    # --- Task 2: Process Albumins ---
    console.rule("[bold]Collecting Albumins[/bold]")
    albumin_pdb_ids = search_pdb_by_text(ALBUMIN_TARGET['search_term'])
    console.print(f"Found {len(albumin_pdb_ids)} structures for '{ALBUMIN_TARGET['search_term']}'.")
    
    albumin_sample_size = min(len(albumin_pdb_ids), ALBUMIN_TARGET['sample_size'])
    selected_albumin_ids = random.sample(albumin_pdb_ids, albumin_sample_size)
    console.print(f"Randomly selected {len(selected_albumin_ids)} structures.")
    
    albumin_records = [{"pdb_id": pid, "group": "albumin"} for pid in selected_albumin_ids]
    process_dataset("Albumins", albumin_records, ALBUMINS_BASE_DIR)

    console.rule("[bold green]All processing complete[/bold green]")

if __name__ == "__main__":
    main()