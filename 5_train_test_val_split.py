import random
from pathlib import Path
import pandas as pd
from rich.console import Console

def create_final_split_report():
    
    RANDOM_SEED = 42
    SPLIT_RATIOS = (0.70, 0.15, 0.15)
    
    REPORT_POSITIVE_CSV = Path("protein_clustering_results/clustered_protein_report.csv")
    REPORT_NEGATIVE_CSV = Path("negative_controls_results/negative_controls_results_report.csv")
    
    OUTPUT_CSV = Path("dataset_split_report.csv")

    console = Console()
    
    if not abs(sum(SPLIT_RATIOS) - 1.0) < 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(RANDOM_SEED)
    all_split_data = []

    reports = {
        "positive": REPORT_POSITIVE_CSV,
        "negative": REPORT_NEGATIVE_CSV,
    }

    console.rule("[bold blue]Generating Final Split Report[/bold blue]")

    for group_name, report_path in reports.items():
        console.print(f"Processing group: [bold cyan]{group_name}[/bold cyan]")
        
        try:
            df = pd.read_csv(report_path)
        except FileNotFoundError:
            console.log(f"[yellow]Warning: Report file not found, skipping: {report_path}[/yellow]")
            continue

        unique_clusters = sorted(df["cluster_id"].unique().tolist())
        rng.shuffle(unique_clusters)

        n_total = len(unique_clusters)
        n_train = int(n_total * SPLIT_RATIOS[0])
        n_val = int(n_total * SPLIT_RATIOS[1])

        train_clusters = set(unique_clusters[:n_train])
        val_clusters = set(unique_clusters[n_train : n_train + n_val])
        
        console.print(f"  - Total clusters: {n_total}")
        console.print(f"  - Assigning to train: {len(train_clusters)} clusters")
        console.print(f"  - Assigning to validation: {len(val_clusters)} clusters")
        console.print(f"  - Assigning to test: {n_total - len(train_clusters) - len(val_clusters)} clusters")

        def assign_split(cluster_id):
            if cluster_id in train_clusters:
                return "train"
            elif cluster_id in val_clusters:
                return "validation"
            else:
                return "test"

        df['split'] = df['cluster_id'].apply(assign_split)
        df['group'] = group_name

        all_split_data.extend(df[['pdb_id', 'cluster_id', 'split', 'group']].to_dict('records'))

    if not all_split_data:
        console.print("[bold red]Error: No data was processed. Check input file paths.[/bold red]")
        return
        
    final_df = pd.DataFrame(all_split_data)
    final_df.sort_values(by=['split', 'group', 'cluster_id', 'pdb_id'], inplace=True)
    
    final_df.to_csv(OUTPUT_CSV, index=False)

    console.rule(f"[bold green]Complete[/bold green]")
    console.print(f"Final report with split assignments saved to: [bold cyan]{OUTPUT_CSV}[/bold cyan]")

    console.print("\n[bold]Split Summary (by number of proteins):[/bold]")
    summary = final_df.groupby(['split', 'group'])['pdb_id'].count().unstack(fill_value=0)
    console.print(summary)
    
if __name__ == "__main__":
    create_final_split_report()