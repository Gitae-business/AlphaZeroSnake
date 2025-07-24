import os
import json
import numpy as np

def json_to_npy(input_folder: str, output_folder: str):
    """
    지정된 폴더의 *.json 파일을 읽어 *_states.npy, *_policies.npy, *_values.npy 파일로 저장
    """
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"Found {len(json_files)} json samples in {input_folder}")

    for json_file in json_files:
        prefix = json_file.replace('.json', '')

        json_path = os.path.join(input_folder, json_file)

        try:
            with open(json_path, 'r') as jf:
                data = json.load(jf)

            states = np.array(data['states'])
            policies = np.array(data['policies'])
            values = np.array(data['values'])

            states_path = os.path.join(output_folder, f"{prefix}_states.npy")
            policies_path = os.path.join(output_folder, f"{prefix}_policies.npy")
            values_path = os.path.join(output_folder, f"{prefix}_values.npy")

            np.save(states_path, states)
            np.save(policies_path, policies)
            np.save(values_path, values)

            print(f"Converted {json_path} → {states_path}, {policies_path}, {values_path}")

        except Exception as e:
            print(f"Failed to convert {json_path}: {e}")


def main():
    input_folder = os.path.abspath("valid")
    output_folder = os.path.abspath("data")

    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    json_to_npy(input_folder, output_folder)

    print("모든 변환이 완료되었습니다.")


if __name__ == "__main__":
    main()
