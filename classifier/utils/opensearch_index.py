import requests
import os
from pathlib import Path
import time

def insert_index(paths: list[str], timeout: int = 30, max_retries: int = 5):
    """
    Mengirim request POST JSON dengan exponential backoff ketika gagal.
    """

    api_url_db = os.getenv("OPENSEARCH_URL", "http://localhost:8155")
    url = f"{api_url_db}/api/add/file/bulk"

    payload = {"file_paths": paths}

    retry = 0
    delay = 1

    while retry <= max_retries:
        try:
            response = requests.post(url, json=payload, timeout=timeout)

            if response.ok:
                return response.text

            if 500 <= response.status_code < 600:
                raise requests.RequestException(f"Server error {response.status_code}")

            return f"Request gagal: {response.status_code} - {response.text}"

        except requests.RequestException as e:
            if retry == max_retries:
                return f"Request gagal setelah {max_retries} retry: {e}"

            time.sleep(delay)
            delay *= 2
            retry += 1

    return "Unknown error"




def list_files_recursive(root_dir: str) -> list[str]:
    """
    Mengambil semua file di dalam root_dir secara rekursif
    dan mengembalikan path dalam format POSIX (/),
    kecuali file PDF.
    """
    result = []
    root_dir = Path(root_dir)

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)

        for f in filenames:
            # cek kalau bukan PDF
            if not f.lower().endswith(".pdf"):
                full_path = (dirpath / f).as_posix()
                result.append(full_path)

    return result


def list_dirs_recursive(root_dir: str) -> list[str]:
    """
    Mengambil semua folder di dalam root_dir secara rekursif
    dan mengembalikan path dalam format POSIX (/).
    """
    result = []
    root_dir = Path(root_dir)

    for dirpath, dirnames, _ in os.walk(root_dir):
        dirpath = Path(dirpath)

        for d in dirnames:
            full_path = (dirpath / d).as_posix()
            result.append(full_path)

    return result


def list_all_recursive(root_dir: str) -> list[str]:
    """
    Mengambil semua path (file dan folder) di dalam root_dir secara rekursif
    dan mengubah path menjadi format POSIX (menggunakan /).
    """
    result = []
    root_dir = Path(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)

        # Tambah folder
        for d in dirnames:
            full_path = (dirpath / d).as_posix()
            result.append(full_path)

        # Tambah file
        for f in filenames:
            full_path = (dirpath / f).as_posix()
            result.append(full_path)

    return result

