#!/usr/bin/env python3
import errno
import os
import requests
import tarfile
from tqdm import tqdm


def download_tar_gz_file(url, target_path, fname, max_retries=10):
    file_path = os.path.join(target_path, fname)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return True

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                with open(file_path, 'wb') as f, tqdm(
                        desc=f"Downloading {fname}",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"Downloaded file: {file_path}")
            return True
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}. Error: {e}")

    print(f"Failed to download {url} after {max_retries} attempts.")
    return False


def extract_tar_gz(file_path, target_path):
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=target_path)
        print(f"Extracted {file_path} to {target_path}")
    except tarfile.TarError as e:
        print(f"Error extracting {file_path}: {e}")


def create_directories(paths):
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory created: {path}")
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Error creating directory {path}: {e}")


def process_datasets(datasets):
    for key, data in datasets.items():
        print(f"Processing {data['url']}")
        if download_tar_gz_file(data['url'], data['target_path'], data['fname']):
            file_path = os.path.join(data['target_path'], data['fname'])
            extract_tar_gz(file_path, data['target_path'])


def main():
    current_path = os.getcwd()
    target_base_path = os.path.join(current_path, 'fvecs')

    create_directories([target_base_path])

    datasets = {
        'sift10K': {
            'url': 'https://github.com/TileDB-Inc/TileDB-Vector-Search/releases/download/0.0.1/siftsmall.tgz',
            'target_path': os.path.join(target_base_path, 'sift10K'),
            'fname': 'siftsmall.tar.gz'
        },
        'sift1M': {
            'url': 'https://figshare.com/ndownloader/files/13755344',
            'target_path': os.path.join(target_base_path, 'sift1M'),
            'fname': 'sift1M.tar.gz'
        }
    }

    create_directories([data['target_path'] for data in datasets.values()])

    process_datasets(datasets)


if __name__ == "__main__":
    main()
