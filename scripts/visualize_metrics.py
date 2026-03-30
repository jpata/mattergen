import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def visualize_metrics(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract entries to find formulas
    entries = data.get('entry', [])
    formulas = []
    for entry in entries:
        comp = entry.get('composition', {})
        formula = "".join([f"{k}{int(v) if v == int(v) else v}" for k, v in comp.items()])
        formulas.append(formula)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        'Formula': formulas,
        'Energy above hull (eV/atom)': data['energy_above_hull_per_atom'],
        'SC Energy above hull (eV/atom)': data['self_consistent_energy_above_hull'],
        'RMSD from relaxation (Å)': data['rmsd_from_relaxation'],
        'Stable': data['stable'],
        'Novel': data['novel'],
        'Unique': data['unique'],
        'Explored': data['is_explored'],
        'Comp Valid': data['comp_validity'],
        'Struct Valid': data['structure_validity'],
        'Fully Valid': data['structure_comp_validity'],
        'Ideal': data['novel_unique_stable']
    })

    # Map boolean values to descriptive strings
    df['Stability Status'] = df['Stable'].map({True: 'Stable', False: 'Unstable'})
    df['Novelty Status'] = df['Novel'].map({True: 'New Structure', False: 'Known Structure'})
    df['SC Stability Status'] = (df['SC Energy above hull (eV/atom)'] <= 0.1).map({True: 'Stable', False: 'Unstable'})

    # Set up the figure with a 3x3 grid
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3)
    sns.set_theme(style="whitegrid")

    # ROW 1: ENERGY & RELAXATION
    # 1. Energy above hull per atom
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=df, x='Formula', y='Energy above hull (eV/atom)', ax=ax1, hue='Stability Status', palette={'Stable': '#2ecc71', 'Unstable': '#e74c3c'})
    ax1.axhline(0.1, ls='--', color='black', alpha=0.5)
    ax1.set_title('Thermodynamic Stability (Reference Hull)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=90, labelsize=8)

    # 2. Self-Consistent Energy above hull
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df, x='Formula', y='SC Energy above hull (eV/atom)', ax=ax2, hue='SC Stability Status', palette={'Stable': '#2ecc71', 'Unstable': '#e74c3c'})
    ax2.axhline(0.1, ls='--', color='black', alpha=0.5)
    ax2.set_title('Self-Consistent Stability (Batch Hull)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=90, labelsize=8)

    # 3. RMSD from relaxation
    ax3 = fig.add_subplot(gs[0, 2])
    sns.barplot(data=df, x='Formula', y='RMSD from relaxation (Å)', ax=ax3, hue='Novelty Status', palette='Set2')
    ax3.set_title('Structural Change (RMSD)', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=90, labelsize=8)

    # ROW 2: DIVERSITY & NOVELTY
    # 4. Ideal Structures (Novel + Unique + Stable)
    ax4 = fig.add_subplot(gs[1, 0])
    ideal_counts = df['Ideal'].value_counts()
    ax4.pie(ideal_counts, labels=ideal_counts.index.map({True: 'Ideal', False: 'Other'}),
            autopct='%1.1f%%', colors=['#9b59b6', '#bdc3c7'], startangle=140)
    ax4.set_title('Ideal Structure Rate\n(New Struct. + Unique + Stable)', fontsize=14, fontweight='bold')

    # 5. Explored Systems
    ax5 = fig.add_subplot(gs[1, 1])
    explored_counts = df['Explored'].value_counts()
    ax5.pie(explored_counts, labels=explored_counts.index.map({True: 'Known Chem. System', False: 'New Chem. System'}), 
            autopct='%1.1f%%', colors=['#3498db', '#e67e22'], startangle=140)
    ax5.set_title('Chemical System Discovery\n(Explored vs Unexplored Systems)', fontsize=14, fontweight='bold')
    # 6. Batch Uniqueness
    ax6 = fig.add_subplot(gs[1, 2])
    unique_counts = df['Unique'].value_counts()
    ax6.pie(unique_counts, labels=unique_counts.index.map({True: 'Unique', False: 'Duplicate'}), 
            autopct='%1.1f%%', colors=['#f1c40f', '#95a5a6'], startangle=140)
    ax6.set_title('Batch Uniqueness\n(Diversity)', fontsize=14, fontweight='bold')

    # ROW 3: VALIDITY
    # 7. Chemical Validity (SMACT)
    ax7 = fig.add_subplot(gs[2, 0])
    comp_counts = df['Comp Valid'].value_counts()
    ax7.pie(comp_counts, labels=comp_counts.index.map({True: 'Valid Chem', False: 'Invalid Chem'}), 
            autopct='%1.1f%%', colors=['#1abc9c', '#7f8c8d'], startangle=140)
    ax7.set_title('Chemical Validity\n(Charge Balance)', fontsize=14, fontweight='bold')

    # 8. Structural Validity
    ax8 = fig.add_subplot(gs[2, 1])
    struct_counts = df['Struct Valid'].value_counts()
    ax8.pie(struct_counts, labels=struct_counts.index.map({True: 'Valid Struct', False: 'Invalid Struct'}), 
            autopct='%1.1f%%', colors=['#34495e', '#ecf0f1'], startangle=140)
    ax8.set_title('Structural Validity\n(Geometric Consistency)', fontsize=14, fontweight='bold')

    # 9. Combined Validity
    ax9 = fig.add_subplot(gs[2, 2])
    full_valid_counts = df['Fully Valid'].value_counts()
    ax9.pie(full_valid_counts, labels=full_valid_counts.index.map({True: 'Fully Valid', False: 'Partially Invalid'}), 
            autopct='%1.1f%%', colors=['#27ae60', '#c0392b'], startangle=140)
    ax9.set_title('Combined Validity\n(Chem + Struct)', fontsize=14, fontweight='bold')

    plt.suptitle('MatterGen Comprehensive Evaluation Dashboard', fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Comprehensive dashboard saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize evaluation metrics")
    parser.add_argument("--json", type=str, default="results/detailed_metrics.json", help="Path to detailed metrics json")
    parser.add_argument("--out", type=str, default="results/detailed_metrics_visualization.png", help="Output image path")
    args = parser.parse_args()

    json_path = Path(args.json)
    output_path = Path(args.out)
    if json_path.exists():
        visualize_metrics(json_path, output_path)
    else:
        print(f"Error: {json_path} not found.")
