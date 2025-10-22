# -*- coding: utf-8 -*-
"""
This script searches for and classifies polyterpene ligands (polyenes)
in the PDB's Chemical Component Dictionary (CCD), with an option to exclude
unwanted ligand codes.

The script performs the following steps:
1.  Uses a blacklist (IGNORE_CODES) to quickly skip irrelevant ligands.
2.  Finds all remaining ligands containing structural motifs of polyterpenes
    using three distinct SMARTS patterns ("head_to_tail", "head_to_head",
    "tail_to_tail").
3.  For each found ligand, it extracts metadata and classifies the match type.
4.  Groups ligand synonyms based on their canonical SMILES representation.
5.  Saves a final report to a CSV file and generates 2D images of the molecules.
6.  Prints a clean, formatted list of unique ligand codes to the console.
"""
import contextlib
import gzip
from pathlib import Path
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from rdkit import RDLogger

import gemmi
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn
)

# --- Configuration ---
CCD_URL = "https://files.rcsb.org/pub/pdb/data/monomers/components.cif.gz"
CCD_FILE = Path("components.cif")
OUTPUT_IMAGE_DIR = Path("polyterpene_ligand_images")
OUTPUT_CSV = Path("polyterpene_ligands_grouped.csv")

# --- BLACKLIST: Add PDB ligand codes here to ignore them during the search ---
IGNORE_CODES = [
    # List of codes to exclude
    "3ZZ", "4C4", "36L", "CYU", "J1R", "BKC", "J1S", "KJL",
    "MXP", "NE6", "NTE", "PUL", "Q6B", "ZH7", "ZHD",
]
# --------------------------------------------------------------------------

# --- SMARTS patterns for different types of terpene linkages ---
PATTERNS = {
    "head_to_tail": Chem.MolFromSmarts("[#6]-[#6](-[CH3])=[#6]-[#6]=[#6]-[#6](-[CH3])=[#6]"),
    "head_to_head": Chem.MolFromSmarts("[#6]-[#6]([CH3])=[#6]-[#6]=[#6]([CH3])-[#6]"),
    "tail_to_tail": Chem.MolFromSmarts("[#6](=[#6]-[CH3])-[#6]-[#6]-[#6](=[#6]-[CH3])"),
}

# --- Global objects ---
console = Console()
BOND_TYPE_MAP = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "AROM": Chem.BondType.AROMATIC,
}

@contextlib.contextmanager
def suppress_rdkit_errors():
    """A context manager to temporarily suppress verbose RDKit logs."""
    logger = RDLogger.logger()
    original_level = logger.level
    logger.setLevel(RDLogger.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original_level)

def download_ccd_file():
    """
    Downloads and decompresses the Chemical Component Dictionary file if it
    does not already exist.
    """
    if CCD_FILE.exists():
        console.print(f"Using existing file: [cyan]{CCD_FILE}[/cyan]")
        return True
    
    console.print(f"Downloading {CCD_URL}...")
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%", "•",
        DownloadColumn(), "•", TransferSpeedColumn(), "•",
        TimeRemainingColumn(),
        console=console,
        transient=True
    )
    try:
        with progress:
            task_id = progress.add_task(f"Downloading {CCD_FILE.name}.gz", total=None)
            response = requests.get(CCD_URL, stream=True, timeout=300)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress.update(task_id, total=total_size)
            
            with gzip.open(response.raw, 'rb') as f_in:
                with open(CCD_FILE, 'wb') as f_out:
                    for chunk in f_in:
                        f_out.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        console.print(f"Successfully downloaded and saved [cyan]{CCD_FILE}[/cyan].")
        return True
    except Exception as e:
        console.print(f"[bold red]Error during download or decompression: {e}[/bold red]")
        if CCD_FILE.exists():
            CCD_FILE.unlink() # Clean up partial file
        return False

def find_canonical_smiles(block: gemmi.cif.Block) -> str | None:
    """Extracts the canonical SMILES string from a CIF block."""
    loop = block.find_loop('_pdbx_chem_comp_descriptor.')
    if not loop:
        return None
    
    descriptor_tags = ['_pdbx_chem_comp_descriptor.descriptor',
                       '_pdbx_chem_comp_descriptor.type',
                       '_pdbx_chem_comp_descriptor.program']
    try:
        table = loop.get_table(descriptor_tags)
        for row in table:
            if row[1] == 'SMILES_CANONICAL' and 'CACTVS' in row[2]:
                return row[0].strip('"')
    except (RuntimeError, IndexError):
        # Handle cases where columns are missing or data is malformed
        pass
    return None

def create_mol_from_cif_block(block: gemmi.cif.Block) -> Chem.Mol | None:
    """Reconstructs an RDKit molecule from atom and bond tables in a CIF block."""
    try:
        atom_ids = block.find_values('_chem_comp_atom.atom_id')
        atom_symbols = block.find_values('_chem_comp_atom.type_symbol')
        
        if not atom_ids:
            return None

        mol = Chem.RWMol()
        atom_map = {name.strip('"'): i for i, name in enumerate(atom_ids)}
        
        for symbol in atom_symbols:
            mol.AddAtom(Chem.Atom(symbol.capitalize()))
        
        bond_atom1s = block.find_values('_chem_comp_bond.atom_id_1')
        if bond_atom1s:
            bond_atom2s = block.find_values('_chem_comp_bond.atom_id_2')
            bond_orders = block.find_values('_chem_comp_bond.value_order')

            for i in range(len(bond_atom1s)):
                idx1 = atom_map.get(bond_atom1s[i].strip('"'))
                idx2 = atom_map.get(bond_atom2s[i].strip('"'))
                bond_type = BOND_TYPE_MAP.get(bond_orders[i].upper(), Chem.BondType.UNSPECIFIED)
                if idx1 is not None and idx2 is not None:
                    mol.AddBond(idx1, idx2, bond_type)
        
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol, catchErrors=True)
        return final_mol
    except Exception:
        return None

def main():
    """Main execution function of the script."""
    if not download_ccd_file():
        return

    ignore_set = set(IGNORE_CODES)
    if ignore_set:
        console.print(f"Ignoring [yellow]{len(ignore_set)}[/yellow] codes from the blacklist.")

    doc = gemmi.cif.read(str(CCD_FILE))
    OUTPUT_IMAGE_DIR.mkdir(exist_ok=True)
    
    found_ligands_data = []
    
    progress = Progress(
        TextColumn("[cyan]Analyzing components..."), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), "•",
        TextColumn("[green]Found: {task.fields[found]}"),
        console=console
    )

    with progress:
        task = progress.add_task("Analysis", total=len(doc), found=0)
        with suppress_rdkit_errors():
            for block in doc:
                ligand_id = block.name
                
                if ligand_id in ignore_set:
                    progress.update(task, advance=1)
                    continue

                mol = None
                smiles_string = find_canonical_smiles(block)
                if smiles_string:
                    mol = Chem.MolFromSmiles(smiles_string)
                
                if not mol:
                    mol = create_mol_from_cif_block(block)
                
                if mol:
                    match_types = [
                        name for name, pattern in PATTERNS.items()
                        if mol.HasSubstructMatch(pattern)
                    ]
                    
                    if match_types:
                        data = {
                            "pdb_code": ligand_id,
                            "match_types": "; ".join(sorted(match_types)),
                            "chemical_name": block.find_value('_chem_comp.name').strip('; \n'),
                            "formula": block.find_value('_chem_comp.formula').strip(),
                            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True)
                        }
                        found_ligands_data.append(data)
                        
                        img_path = OUTPUT_IMAGE_DIR / f"{ligand_id}.png"
                        try:
                            MolToImage(mol, size=(300, 300), kekulize=True).save(img_path)
                        except Exception:
                            pass
                
                progress.update(task, advance=1, found=len(found_ligands_data))

    if not found_ligands_data:
        console.print("[yellow]No polyterpene ligands found (after applying filters).[/yellow]")
        return
        
    df = pd.DataFrame(found_ligands_data)
    df['group_id'] = df.groupby('canonical_smiles')['pdb_code'].transform('first')
    df = df.sort_values(by=['group_id', 'pdb_code'])
    
    cols_order = ['group_id', 'pdb_code', 'match_types', 'chemical_name', 'formula', 'canonical_smiles']
    df = df[cols_order]
    
    df.to_csv(OUTPUT_CSV, index=False)
    
    console.print(f"\nFound [bold]{len(df)}[/bold] PDB codes corresponding to "
                  f"[bold]{df['group_id'].nunique()}[/bold] unique polyterpene molecules.")
    console.print(f"Detailed report saved to: [bold cyan]{OUTPUT_CSV}[/bold cyan]")
    console.print(f"Molecule images saved in: [bold cyan]{OUTPUT_IMAGE_DIR}[/bold cyan]")
    
    synonym_groups = [group for _, group in df.groupby('group_id') if len(group) > 1]
    if synonym_groups:
        console.print("\n[bold]Synonyms found (different codes for the same molecule):[/bold]")
        for group in synonym_groups:
            codes = group['pdb_code'].tolist()
            console.print(f"  - Group [bold green]{codes[0]}[/bold green]: {', '.join(codes)}")
    
    unique_codes = sorted(df['group_id'].unique())
    console.print(f"\n[bold]Final list of {len(unique_codes)} unique codes for subsequent use:[/bold]")
    formatted_list = "LIGAND_CODES = [\n"
    for i in range(0, len(unique_codes), 10):
        chunk = unique_codes[i:i + 10]
        formatted_list += f"    {', '.join(f'\"{code}\"' for code in chunk)},\n"
    formatted_list = formatted_list.rstrip(',\n') + "\n]"
    console.print(formatted_list, highlight=False)

if __name__ == "__main__":
    main()