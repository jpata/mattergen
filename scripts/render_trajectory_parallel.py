import os
import zipfile
import tempfile
import argparse
import shutil
import imageio
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from ase.io import read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

def render_frame(args):
    """Worker function to render a single frame to a PNG file with fixed camera/view"""
    i, atoms, tmp_dir, xlim, ylim, rotation = args
    # Create a 1920x1080 figure (19.2 x 10.8 inches at 100 DPI)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    # Add axes that fill the entire figure for full control
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Render with fixed rotation
    plot_atoms(atoms, ax, radii=0.3, rotation=rotation)
    
    # Apply fixed axes limits to stabilize the camera and ensure centering
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    
    # Overlay Title
    fig.text(0.5, 0.95, f"Step {i} | {atoms.get_chemical_formula()}", 
             ha='center', va='top', fontsize=24, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=10))
    
    frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)
    return frame_path

def render_trajectory_parallel(zip_path, trajectory_filename, output_path, fps=50, num_workers=None):
    print(f"Extracting {trajectory_filename} from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        if trajectory_filename not in z.namelist():
            print(f"Error: {trajectory_filename} not found in {zip_path}")
            return
        with z.open(trajectory_filename) as f:
            with tempfile.NamedTemporaryFile(suffix='.extxyz', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                frames = read(tmp_path, index=':')
            finally:
                os.remove(tmp_path)

    num_frames = len(frames)
    rotation = '45x,45y,0z'
    
    print(f"Calculating global bounds for stable camera...")
    # To stabilize the camera, we need to find the max/min bounds across all frames
    # after applying the projection/rotation.
    all_x = []
    all_y = []
    
    # We do a quick dry-run of plot_atoms to see what limits it would choose
    # and then take the global min/max.
    for i in range(0, num_frames, max(1, num_frames // 20)): # Sample frames to save time
        fig_tmp, ax_tmp = plt.subplots()
        plot_atoms(frames[i], ax_tmp, rotation=rotation)
        cur_xlim = ax_tmp.get_xlim()
        cur_ylim = ax_tmp.get_ylim()
        all_x.extend(cur_xlim)
        all_y.extend(cur_ylim)
        plt.close(fig_tmp)
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Target 16:9 aspect ratio
    target_aspect = 1920 / 1080
    if width / height > target_aspect:
        display_width = width
        display_height = width / target_aspect
    else:
        display_height = height
        display_width = height * target_aspect
    
    # Add 20% padding to ensure atoms aren't cut off
    display_width *= 1.2
    display_height *= 1.2
    
    global_xlim = (center_x - display_width / 2, center_x + display_width / 2)
    global_ylim = (center_y - display_height / 2, center_y + display_height / 2)

    print(f"Loaded {num_frames} frames. Rendering in parallel using {num_workers or os.cpu_count()} workers...")

    tmp_frame_dir = tempfile.mkdtemp()
    
    try:
        worker_args = [(i, frames[i], tmp_frame_dir, global_xlim, global_ylim, rotation) for i in range(num_frames)]
        
        with Pool(processes=num_workers) as pool:
            frame_paths = list(tqdm(pool.imap(render_frame, worker_args), total=num_frames, desc="Rendering frames"))

        print(f"Stitching frames into video: {output_path}")
        
        with imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p', quality=8) as writer:
            for path in tqdm(frame_paths, desc="Stitching frames"):
                image = imageio.imread(path)
                writer.append_data(image)
        
        print(f"Video saved successfully to {output_path}")

    finally:
        shutil.rmtree(tmp_frame_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Parallel Render MatterGen trajectory to video")
    parser.add_argument("--zip", type=str, default="results/generated_trajectories.zip", help="Path to trajectories zip")
    parser.add_argument("--name", type=str, default="gen_0.extxyz", help="Trajectory filename in zip")
    parser.add_argument("--out", type=str, default="results/trajectory_render_parallel.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")

    args = parser.parse_args()
    render_trajectory_parallel(args.zip, args.name, args.out, args.fps, args.workers)
