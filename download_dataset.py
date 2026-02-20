import gdown
import argparse
import os
import zipfile
import shutil

# Google Drive 파일 ID (ZIP 파일)
zip_file_id = "1t9PUYuHsUa_cHxJTkEDP6tpMLNjuaSJ8"  # ZIP 파일의 실제 ID로 변경하세요
zip_file_name = "data.zip"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="./data")
    args = parser.parse_args()

    # 데이터 저장 디렉토리 생성
    os.makedirs(args.path, exist_ok=True)

    # ZIP 파일 다운로드
    zip_path = os.path.join(args.path, zip_file_name)
    gdown.download(f'https://drive.google.com/uc?id={zip_file_id}', zip_path, quiet=False)

    # ZIP 파일 압축 해제
    extract_path = os.path.join(args.path, "unzipped")
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 다운로드된 ZIP 파일 삭제 (선택 사항)
    os.remove(zip_path)

    print(f"Dataset extracted to {extract_path}")

