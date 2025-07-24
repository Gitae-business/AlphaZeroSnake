import os
import json
import shutil

# 입력 상대 경로
DOWNLOAD_REL_PATH = './legacy/played'
VALID_REL_PATH = './valid'

# 절대경로 변환
DOWNLOAD_DIR = os.path.abspath(DOWNLOAD_REL_PATH)
VALID_DIR = os.path.abspath(VALID_REL_PATH)

# 필수 필드 리스트
REQUIRED_FIELDS = ['values', 'states', 'policies']

def is_valid_json(data):
    if not isinstance(data, dict):
        return False
    return all(field in data for field in REQUIRED_FIELDS)

def validate_and_move_files():
    if not os.path.exists(VALID_DIR):
        os.makedirs(VALID_DIR)

    files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.json')]
    valid_count = 0
    invalid_count = 0

    for filename in files:
        path = os.path.join(DOWNLOAD_DIR, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if is_valid_json(data):
                # valid 폴더로 이동
                shutil.move(path, os.path.join(VALID_DIR, filename))
                valid_count += 1
            else:
                print(f'필수 필드 누락: {filename}')
                invalid_count += 1
        except Exception as e:
            print(f'파일 읽기/파싱 실패: {filename} - {e}')
            invalid_count += 1

    print(f'검증 완료: 유효 {valid_count}개, 무효 {invalid_count}개')

if __name__ == '__main__':
    validate_and_move_files()
