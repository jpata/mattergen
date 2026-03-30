import os
import zipfile
import tempfile
import argparse
import shutil
import subprocess
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
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Render with fixed rotation
    plot_atoms(atoms, ax, radii=0.3, rotation=rotation)
    
    # Apply fixed axes limits to stabilize the camera
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Fixed Title position and formatting
    # Using ax.text for more stable positioning if needed, 
    # but with fixed limits, ax.set_title should be fine.
    ax.set_title(f"Step {i} | {atoms.get_chemical_formula()}", pad=20)
    ax.axis('off')
    
    frame_path = os.path.join(tmp_dir, f"frame_{i:06d}.png")
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
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
    
    global_xlim = (min(all_x), max(all_x))
    global_ylim = (min(all_y), max(all_y))
    
    # Add some padding
    x_pad = (global_xlim[1] - global_xlim[0]) * 0.1
    y_pad = (global_ylim[1] - global_ylim[0]) * 0.1
    global_xlim = (global_xlim[0] - x_pad, global_xlim[1] + x_pad)
    global_ylim = (global_ylim[0] - y_pad, global_ylim[1] + y_pad)

    print(f"Loaded {num_frames} frames. Rendering in parallel using {num_workers or os.cpu_count()} workers...")

    tmp_frame_dir = tempfile.mkdtemp()
    
    try:
        worker_args = [(i, frames[i], tmp_frame_dir, global_xlim, global_ylim, rotation) for i in range(num_frames)]
        
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(render_frame, worker_args), total=num_frames, desc="Rendering frames"))

        print(f"Stitching frames into video: {output_path}")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmp_frame_dir, 'frame_%06d.png'),
            '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg Error:")
            print(result.stderr)
        else:
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
