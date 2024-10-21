#!/usr/bin/env python3
import os
import requests
import errno
from tqdm import tqdm

def download_hd5_file(url, target_path, fname, max_retries=10):
    file_path = os.path.join(target_path, fname)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return True

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as f, tqdm(
                desc=f"Downloading {fname}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"Downloaded file: {file_path}")
            return True
        except requests.RequestException as e:
            print(f"\nAttempt {attempt + 1} failed for {url}. Error: {e}")

    print(f"Failed to download {url} after {max_retries} attempts.")
    return False

def create_directories(paths):
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory created: {path}")
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Error creating directory {path}: {e}")

def main():
    current_path = os.getcwd()
    target_base_path = os.path.join(current_path, 'hdf5')

    create_directories([target_base_path])

    datasets = {
        'enron': 'https://www.dropbox.com/s/z56uf5qdmpp6iqo/enron.hdf5?dl=1',
        'sun': 'https://www.dropbox.com/s/h8lvtvfbejghi99/sun.hdf5?dl=1',
        'trevi': 'https://www.dropbox.com/s/9ezi2gkuhnkem6d/trevi.hdf5?dl=1',
        'glove': 'https://www.dropbox.com/s/xg0jvdnp8oszhuu/glove.hdf5?dl=1',
        'msong': 'https://www.dropbox.com/s/mh11y5q7dugehwi/millionSong.hdf5?dl=1'
    }

    target_paths = {key: os.path.join(target_base_path, key) for key in datasets.keys()}

    create_directories(target_paths.values())

    for key, url in datasets.items():
        download_hd5_file(url, target_paths[key], f"{key}.hdf5")

if __name__ == "__main__":
    main()