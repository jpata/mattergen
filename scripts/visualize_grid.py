import os
import zipfile
import tempfile
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms
from math import ceil, sqrt
from pathlib import Path

def visualize_grid(zip_path, metrics_path, output_path, rotation='45x,45y,0z'):
    print(f"Opening {zip_path} and {metrics_path}...")
    
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    with zipfile.ZipFile(zip_path, 'r') as z:
        filenames = [f for f in z.namelist() if f.endswith('.extxyz')]
        filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0])) # Sort by number: gen_0, gen_1...
        
        num_structures = len(filenames)
        if num_structures == 0:
            print("No .extxyz files found in the zip.")
            return

        # Determine grid size
        cols = ceil(sqrt(num_structures))
        rows = ceil(num_structures / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
        if num_structures == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        print(f"Extracting final steps and metrics for {num_structures} trajectories...")
        
        for i, filename in enumerate(filenames):
            ax = axes[i]
            # Match entry_id with the index (assuming gen_i corresponds to entry_id i)
            # Find the entry index where entry_id == i
            entry_idx = None
            for idx, entry in enumerate(metrics_data['entry']):
                if entry.get('entry_id') == i:
                    entry_idx = idx
                    break
            
            if entry_idx is None:
                print(f"Warning: Could not find metrics for {filename}")
                continue

            # Extract metrics for this entry
            e_hull = metrics_data['energy_above_hull_per_atom'][entry_idx]
            rmsd = metrics_data['rmsd_from_relaxation'][entry_idx]
            stable = metrics_data['stable'][entry_idx]
            novel = metrics_data['novel'][entry_idx]
            
            with z.open(filename) as f:
                with tempfile.NamedTemporaryFile(suffix='.extxyz', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    atoms = read(tmp_path, index='-1')
                    formula = atoms.get_chemical_formula()
                    
                    plot_atoms(atoms, ax, radii=0.3, rotation=rotation)
                    
                    # Create detailed title
                    title_text = (
                        f"{filename} | {formula}\n"
                        f"E_hull: {e_hull:.3f} eV/atom\n"
                        f"RMSD: {rmsd:.3f} Å\n"
                        f"Status: {'Stable' if stable else 'Unstable'} | {'New Struct.' if novel else 'Known Struct.'}"
                    )
                    
                    ax.set_title(title_text, fontsize=9, pad=10, fontweight='bold' if stable else 'normal', color='red' if (stable and novel) else 'black')
                    ax.axis('off')
                finally:
                    os.remove(tmp_path)
        
        # Turn off axes for empty grid cells
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Detailed grid visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render final step and metrics of all trajectories on a grid")
    parser.add_argument("--zip", type=str, default="results/generated_trajectories.zip", help="Path to trajectories zip")
    parser.add_argument("--metrics", type=str, default="results/detailed_metrics.json", help="Path to detailed metrics json")
    parser.add_argument("--out", type=str, default="results/final_structures_grid_detailed.png", help="Output image path")
    parser.add_argument("--rotation", type=str, default="45x,45y,0z", help="Rotation for plotting")

    args = parser.parse_args()
    
    if Path(args.zip).exists() and Path(args.metrics).exists():
        visualize_grid(args.zip, args.metrics, args.out, args.rotation)
    else:
        print("Error: Zip or Metrics file not found.")
