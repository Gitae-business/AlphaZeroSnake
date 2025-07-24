
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from collections import defaultdict

def visualize_game(states, output_path):
    """
    주어진 상태(states) 배열을 기반으로 게임 플레이를 시각화하고 mp4 파일로 저장합니다.
    개선된 시각화: 뱀 길이 표시, 사용자 지정 색상 유지.
    """
    fig, ax = plt.subplots()
    board_height = states.shape[3]
    board_width = states.shape[2]

    # Pre-calculate snake lengths for each frame and find the max length
    snake_lengths = []
    for frame_idx in range(len(states)):
        head_count = np.sum(states[frame_idx, 0, :, :])
        body_count = np.sum(states[frame_idx, 1, :, :])
        snake_lengths.append(int(head_count + body_count))
    max_length = max(snake_lengths) if snake_lengths else 0

    # Adjust layout to make space for the progress bar at the bottom
    fig.subplots_adjust(bottom=0.2)

    # Create a new axes for the progress bar
    pbar_ax = fig.add_axes([0.125, 0.05, 0.775, 0.04])

    def update(frame):
        ax.clear()
        state = states[frame]
        current_length = snake_lengths[frame]

        # --- Main Grid --- 
        ax.set_xlim(-0.5, board_width - 0.5)
        ax.set_ylim(-0.5, board_height - 0.5)
        ax.set_xticks(np.arange(board_width))
        ax.set_yticks(np.arange(board_height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, color='black', linewidth=0.5, alpha=0.5)
        
        # Display snake length at the top
        ax.set_title(f"Length: {current_length} / {max_length}")

        # --- Draw elements (respecting user's color choices) ---
        # Channel 2: Food (Red)
        food_channel = state[2, :, :]
        food_pos = np.argwhere(food_channel == 1)
        for pos in food_pos:
            rect = patches.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='red')
            ax.add_patch(rect)

        # Channel 1: Player's snake body (Green - as per user change)
        body_channel = state[1, :, :]
        body_pos = np.argwhere(body_channel == 1)
        for pos in body_pos:
            rect = patches.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='green')
            ax.add_patch(rect)

        # Channel 0: Player's snake head (Green - as per user change)
        head_channel = state[0, :, :]
        head_pos = np.argwhere(head_channel == 1)
        for pos in head_pos:
            rect = patches.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, linewidth=1, edgecolor='none', facecolor='green')
            ax.add_patch(rect)

        ax.invert_yaxis()

        # --- Progress Bar --- 
        pbar_ax.clear()
        pbar_ax.set_xlim(0, 1)
        pbar_ax.set_ylim(0, 1)
        pbar_ax.axis('off')

        pbar_ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='lightgray', transform=pbar_ax.transAxes, clip_on=False))
        progress = (frame + 1) / len(states)
        pbar_ax.add_patch(patches.Rectangle((0, 0), progress, 1, facecolor='royalblue', transform=pbar_ax.transAxes, clip_on=False))
        pbar_ax.text(0.5, 0.5, f'{frame + 1} / {len(states)}', ha='center', va='center', color='white', fontsize=8, transform=pbar_ax.transAxes)

    ani = animation.FuncAnimation(fig, update, frames=len(states), interval=50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        ani.save(output_path, writer='ffmpeg', fps=10)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure ffmpeg is installed and in your system's PATH.")

    plt.close(fig)

def main(data_dir, output_dir):
    """
    data_dir에서 npy 파일들을 읽어, value별로 정렬하고,
    각 value 그룹에서 길이가 가장 짧은 게임을 선택하여 시각화합니다.
    """
    value_files = glob.glob(os.path.join(data_dir, "*_values.npy"))

    if not value_files:
        print(f"No data files found in {data_dir}")
        return

    games_by_value = defaultdict(list)
    for vf in value_files:
        try:
            values = np.load(vf)
            avg_value = int(np.mean(values))
            prefix = vf.replace("_values.npy", "")
            
            states_path = f"{prefix}_states.npy"
            if os.path.exists(states_path):
                states = np.load(states_path)
                game_length = len(states)
                games_by_value[avg_value].append((prefix, game_length))

        except Exception as e:
            print(f"Could not process {vf}: {e}")

    sorted_values = sorted(games_by_value.keys())

    for value in sorted_values:
        if not games_by_value[value]:
            continue

        shortest_game_prefix, min_length = min(games_by_value[value], key=lambda x: x[1])
        states_path = f"{shortest_game_prefix}_states.npy"

        try:
            states = np.load(states_path)

            if states.ndim != 4 or states.shape[1] != 3:
                print(f"Skipping {states_path} due to unexpected shape: {states.shape}")
                continue

            output_filename = f"game_value_{value}_shortest_len_{min_length}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Visualizing shortest game for value {value} (len: {min_length}) from {os.path.basename(shortest_game_prefix)}")
            visualize_game(states, output_path)

        except Exception as e:
            print(f"Failed to visualize {states_path}: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Snake game replays from .npy files.")
    parser.add_argument("data_dir", type=str, help="Directory containing the .npy game data files.")
    parser.add_argument("output_dir", type=str, help="Directory where the output .mp4 files will be saved.")
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
