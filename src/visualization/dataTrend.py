import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import argparse
from collections import namedtuple

# A simple data structure for our samples
GameSample = namedtuple("GameSample", ["timestamp", "avg_value", "prefix"])

def load_and_sort_data(data_dir):
    """
    Loads game data from the specified directory, extracts metadata,
    and sorts it by timestamp (oldest first).
    """
    value_files = glob.glob(os.path.join(data_dir, "*_values.npy"))
    if not value_files:
        print(f"No data files found in {data_dir}")
        return []

    samples = []
    # Regex to extract the timestamp from the filename
    timestamp_regex = re.compile(r'_(\d{8}_\d{6}_\d{6})_')

    for vf in value_files:
        match = timestamp_regex.search(os.path.basename(vf))
        if not match:
            print(f"Warning: Could not extract timestamp from {os.path.basename(vf)}. Skipping.")
            continue
        
        timestamp = match.group(1)
        prefix = vf.replace("_values.npy", "")
        
        try:
            values = np.load(vf)
            avg_value = int(np.mean(values))
            samples.append(GameSample(timestamp, avg_value, prefix))
        except Exception as e:
            print(f"Could not process {vf}: {e}")

    samples.sort(key=lambda x: x.timestamp)
    return samples

def create_trend_animation(all_samples, output_path):
    """
    Creates an animation showing the trend of the top 100 game values over time.
    """
    INITIAL_POOL_SIZE = 100
    STEP_SIZE = 20

    if len(all_samples) < INITIAL_POOL_SIZE:
        print("Not enough data to start the animation (requires at least 100 samples).")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    num_frames = (len(all_samples) - INITIAL_POOL_SIZE) // STEP_SIZE + 1

    def update(frame):
        ax.clear()

        end_index = min(INITIAL_POOL_SIZE + frame * STEP_SIZE, len(all_samples))
        current_pool = all_samples[:end_index]

        current_pool.sort(key=lambda x: x.avg_value, reverse=True)
        top_100_samples = current_pool[:INITIAL_POOL_SIZE]
        
        values_to_plot = [s.avg_value for s in top_100_samples]

        if not values_to_plot:
            ax.set_title(f"Distribution of Top 100 Game Values (Step {frame + 1}/{num_frames})\n(Considering {len(current_pool)} Oldest Files)")
            ax.text(0.5, 0.5, "No data to display", ha='center', va='center', transform=ax.transAxes)
            return

        # Dynamically calculate x-axis range for the current frame's data
        local_x_min = min(values_to_plot) - 2
        local_x_max = max(values_to_plot) + 2

        # Create the histogram with dynamic bins
        ax.hist(values_to_plot, bins=range(local_x_min, local_x_max + 1), align='left', rwidth=0.8, color='skyblue', edgecolor='black')

        ax.set_title(f"Distribution of Top 100 Game Values (Step {frame + 1}/{num_frames})\n(Considering {len(current_pool)} Oldest Files)")
        ax.set_xlabel("Game Value (Snake Length)")
        ax.set_ylabel("Number of Games")
        
        # Set dynamic x-axis limits for the current frame
        ax.set_xlim(local_x_min, local_x_max)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500)

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        ani.save(output_path, writer='ffmpeg', fps=2)
        print(f"Data trend animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure ffmpeg is installed and in your system's PATH.")

    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate a video showing the trend of game data values.")
    parser.add_argument("data_dir", type=str, help="Directory containing the game data files (e.g., legacyData).")
    parser.add_argument("output_file", type=str, help="Path to save the output .mp4 file.")
    args = parser.parse_args()

    print(f"Loading and sorting data from {args.data_dir}...")
    all_samples = load_and_sort_data(args.data_dir)
    
    if all_samples:
        print(f"Found {len(all_samples)} total samples.")
        create_trend_animation(all_samples, args.output_file)

if __name__ == '__main__':
    main()
