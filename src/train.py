import os
import sys
import ast
import signal
import numpy as np
from glob import glob
import multiprocessing
from functools import partial
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from main import SelfPlay
from model.network import AlphaZeroNet

pool = None
dataloader_iter = None

# Define a custom Dataset for loading training data
class SnakeDataset(Dataset):
    def __init__(self, data_dir, board_width=10, board_height=10):
        self.samples = []
        self.max_length = board_width * board_height
        npy_files = glob(os.path.join(data_dir, "*_states.npy"))

        for file_path in npy_files:
            prefix = file_path.replace("_states.npy", "")

            try:
                states = self.load_and_validate(f"{prefix}_states.npy")
                policies = self.load_and_validate(f"{prefix}_policies.npy")
                values = self.load_and_validate(f"{prefix}_values.npy")
            except Exception as e:
                print(f"Skipping invalid file set: {prefix}, reason: {e}")
                continue

            if len(states) != len(policies) or len(states) != len(values):
                print(f"Skipping inconsistent lengths in: {prefix}")
                continue

            for s, p, v in zip(states, policies, values):
                s_tensor = torch.tensor(s, dtype=torch.float32)
                p_tensor = torch.tensor(p, dtype=torch.float32)
                v_normalized = float(v) / self.max_length
                v_tensor = torch.tensor(v_normalized, dtype=torch.float32)

                self.samples.append((s_tensor, p_tensor, v_tensor))

    def load_and_validate(self, path):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, str):
            arr = ast.literal_eval(arr)
            arr = np.array(arr, dtype=np.float32)

        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{path} is not a numpy array")
        if arr.ndim < 1:
            raise ValueError(f"{path} has invalid shape: {arr.shape}")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{path} contains non-numeric data")
        return arr
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_loss(model, dataset, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for states, policies, values in dataloader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            log_ps, predicted_values = model(states)
            policy_loss = -torch.sum(policies * log_ps, dim=1).mean()
            value_loss = F.mse_loss(predicted_values.squeeze(), values)
            NORM_FACTOR = 0.0008
            loss = policy_loss * NORM_FACTOR + value_loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_model(model, dataset, num_epochs=10, batch_size=64, learning_rate=0.001, model_save_path="best_model.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path} to check its loss.")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        best_loss = evaluate_loss(model, dataset, batch_size)
        print(f"Existing model loss: {best_loss:.6f}")
    else:
        best_loss = float('inf')
        print("No existing model found. Starting fresh.")

    model.train()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0

        for batch_idx, (states, policies, values) in enumerate(dataloader):
            states, policies, values = states.to(device), policies.to(device), values.to(device)

            optimizer.zero_grad()
            log_ps, predicted_values = model(states)

            policy_loss = -torch.sum(policies * log_ps, dim=1).mean()
            value_loss = F.mse_loss(predicted_values.squeeze(), values)

            NORM_FACTOR = 0.0008
            loss = policy_loss * NORM_FACTOR + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}, Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = policy_loss_sum / len(dataloader)
        avg_value_loss = value_loss_sum / len(dataloader)

        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.6f}, "
              f"Avg Policy Loss: {avg_policy_loss:.6f}, Avg Value Loss: {avg_value_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with improved loss: {best_loss:.6f}")

    return model

def run_self_play(BOARD_WIDTH, BOARD_HEIGHT, NUM_ACTIONS, NUM_SNAKES, MODEL_STATE_DICT, DATA_DIR, iteration, game_idx):
    """
    프로세스별로 실행되는 함수. SelfPlay를 생성하고 한 판 실행 후 npy 저장.
    """
    input_shape = (3, BOARD_WIDTH, BOARD_HEIGHT)
    model = AlphaZeroNet(input_shape, NUM_ACTIONS)
    model.load_state_dict(MODEL_STATE_DICT)
    model.eval()

    self_play_agent = SelfPlay(BOARD_WIDTH, BOARD_HEIGHT, NUM_ACTIONS, NUM_SNAKES, model=model)
    game_data = self_play_agent.play_game()

    # Convert to numpy arrays for efficient saving
    states_np = np.array([item["state"] for item in game_data])
    policies_np = np.array([item["policy"] for item in game_data])
    values_np = np.array([item["value"] for item in game_data])
    avg_value = values_np.mean()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name_prefix = f"game_data_iter{iteration+1}_game{game_idx+1}_{timestamp}"

    np.save(os.path.join(DATA_DIR, f"{file_name_prefix}_states.npy"), states_np)
    np.save(os.path.join(DATA_DIR, f"{file_name_prefix}_policies.npy"), policies_np)
    np.save(os.path.join(DATA_DIR, f"{file_name_prefix}_values.npy"), values_np)

    print(f"  [PID {os.getpid()}] Game {game_idx+1} data ({len(game_data)} samples, value: {int(avg_value)}) saved to {file_name_prefix}_*.npy")
    return True

def maintain_data(DATA_DIR, LEGACY_DIR="legacyData", MAX_SAMPLES=500):
    """
    value가 높은 데이터부터 MAX_SAMPLES개 유지하고, 나머지는 legacy로 이동
    """
    os.makedirs(LEGACY_DIR, exist_ok=True)

    # 모든 샘플의 prefix를 수집
    value_files = sorted(glob(os.path.join(DATA_DIR, "*_values.npy")))

    # value별로 기록
    samples = []
    for vf in value_files:
        values = np.load(vf)
        avg_value = np.mean(values)              # 평균 가치
        sample_length = len(values)              # 스텝 수
        prefix = vf.replace("_values.npy", "")
        samples.append( (avg_value, sample_length, prefix) )

    # 1차: avg_value 내림차순, 2차: sample_length 오름차순
    samples.sort(key=lambda x: (-x[0], x[1]))

    # 유지되는 데이터
    kept_samples = samples[:MAX_SAMPLES]
    removed_samples = samples[MAX_SAMPLES:]

    # legacy로 이동
    for _, _, prefix in removed_samples:
        for suffix in ["_states.npy", "_policies.npy", "_values.npy"]:
            src = f"{prefix}{suffix}"
            dst = os.path.join(LEGACY_DIR, os.path.basename(src))
            if os.path.exists(src):
                os.rename(src, dst)

    # 통계 출력
    print(f"Kept {len(kept_samples)} samples.")
    print_value_distribution(kept_samples)

def print_value_distribution(kept_samples):
    """
    유지된 데이터의 value 분포를 오름차순으로 출력한다.
    """
    if not kept_samples:
        print("No samples to display.")
        return

    # value 별 개수 집계
    from collections import Counter
    values = [int(s[0]) for s in kept_samples]
    counter = Counter(values)

    # 오름차순 정렬
    sorted_items = sorted(counter.items())

    print("\n--- Value Distribution of Kept Samples ---")
    for value, count in sorted_items:
        print(f"Value {value:>5}: {count} samples")
    print("------------------------------------------\n")

def generate_self_play_data(iteration, games_per_iteration, model_state_dict, data_dir,
                            board_width=10, board_height=10, num_actions=4, num_snakes=1, num_workers=4):
    global pool
    print(f"\n--- Data Generation Iteration {iteration + 1} ---")
    print(f"Generating {games_per_iteration} self-play games in parallel...")

    if pool is None:
        pool = multiprocessing.Pool(processes=num_workers)

    args_func = partial(
        run_self_play,
        board_width, board_height, num_actions, num_snakes,
        model_state_dict, data_dir, iteration
    )

    pool.map(args_func, range(games_per_iteration))

def train_model_iteration(iteration, model, model_save_path, data_dir, num_epochs=10):
    print("\n--- Training Phase ---")
    print("Loading all generated data for training...")
    dataset = SnakeDataset(data_dir)
    print(f"Loaded {len(dataset)} samples for training.")

    if len(dataset) == 0:
        print("No training data found after data generation. Skipping training for this iteration.")
        return model

    print("Starting model training...")
    model = train_model(model, dataset, num_epochs=num_epochs, batch_size=64, learning_rate=0.001,
                        model_save_path=model_save_path)
    print("Training complete for this iteration.")
    return model

def cleanup():
    global pool, dataloader_iter
    print("Cleaning up resources...")

    # Pool 종료
    if pool is not None:
        print("Terminating multiprocessing pool...")
        pool.terminate()
        pool.join()
        pool = None

    # DataLoader iterator 종료 (if needed)
    if dataloader_iter is not None:
        print("Closing dataloader iterator...")
        try:
            dataloader_iter._shutdown_workers()
        except Exception as e:
            print(f"Error shutting down dataloader workers: {e}")
        dataloader_iter = None

    print("Cleanup done. Exiting.")
    sys.exit(0)

def signal_handler(sig, frame):
    print(f"Signal {sig} received. Exiting gracefully.")
    cleanup()

def main():
    signal.signal(signal.SIGINT, signal_handler)    # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)   # kill 명령 등

    DATA_DIR = "data"
    MODEL_SAVE_PATH = r"E:\workspace\Python\SelfPlaySnake\trained_model.pth"
    os.makedirs(DATA_DIR, exist_ok=True)

    BOARD_WIDTH = 10
    BOARD_HEIGHT = 10
    NUM_ACTIONS = 4
    NUM_SNAKES = 1

    GAMES_PER_ITERATION = 12
    NUM_WORKERS = 6
    NUM_EPOCHS = 4
    NUM_TRAINING_ITERATIONS = 10

    # Initialize model
    input_shape = (3, BOARD_WIDTH, BOARD_HEIGHT)
    current_model = AlphaZeroNet(input_shape, NUM_ACTIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model.to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading pre-existing model from {MODEL_SAVE_PATH}")
        current_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        current_model.eval()

    MODEL_STATE_DICT = current_model.state_dict()

    for iteration in range(NUM_TRAINING_ITERATIONS):
        # # 데이터 생성
        generate_self_play_data(
            iteration, GAMES_PER_ITERATION, MODEL_STATE_DICT, DATA_DIR,
            board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT, num_actions=NUM_ACTIONS,
            num_snakes=NUM_SNAKES, num_workers=NUM_WORKERS
        )

        # 데이터 유지
        maintain_data(DATA_DIR, LEGACY_DIR="legacyData", MAX_SAMPLES=100)

        # 학습
        current_model = train_model_iteration(iteration, current_model, MODEL_SAVE_PATH, DATA_DIR, num_epochs=NUM_EPOCHS)

    print("\n--- All training iterations complete ---")
    print(f"Final trained model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()