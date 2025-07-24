import os
import json
import numpy as np

def npy_to_json(input_folder: str, output_folder: str):
    """
    지정된 폴더의 *_states.npy, *_policies.npy, *_values.npy 파일을 json으로 변환하여 저장
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    npy_files = [f for f in os.listdir(input_folder) if f.endswith('_states.npy')]
    print(f"Found {len(npy_files)} game samples in {input_folder}")

    for states_file in npy_files:
        prefix = states_file.replace('_states.npy', '')

        states_path = os.path.join(input_folder, f"{prefix}_states.npy")
        policies_path = os.path.join(input_folder, f"{prefix}_policies.npy")
        values_path = os.path.join(input_folder, f"{prefix}_values.npy")

        if not (os.path.exists(states_path) and os.path.exists(policies_path) and os.path.exists(values_path)):
            print(f"Missing files for {prefix}, skipping.")
            continue

        states = np.load(states_path, allow_pickle=True).tolist()
        policies = np.load(policies_path, allow_pickle=True).tolist()
        values = np.load(values_path, allow_pickle=True).tolist()

        json_data = {
            "states": states,
            "policies": policies,
            "values": values
        }

        json_filename = f"{prefix}.json"
        json_path = os.path.join(output_folder, json_filename)

        with open(json_path, 'w') as jf:
            json.dump(json_data, jf)

        print(f"Converted {prefix} → {json_path}")


def main():
    input_folder = os.path.abspath("data")
    output_folder = os.path.abspath("temp")

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    npy_to_json(input_folder, output_folder)

    print("모든 변환이 완료되었습니다.")


if __name__ == "__main__":
    main()
