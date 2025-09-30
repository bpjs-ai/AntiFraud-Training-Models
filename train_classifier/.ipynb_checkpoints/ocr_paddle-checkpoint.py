import os
import csv
from paddleocr import PaddleOCR
from tqdm import tqdm
from PIL import Image
import numpy as np
import gc
import paddle
import multiprocessing as mp
import argparse

import warnings
warnings.filterwarnings("ignore")


def preprocess_image(image_path, max_size=1024):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        if original_width > max_size or original_height > max_size:
            if original_width > original_height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        return np.array(img)


def worker_extract(args):
    image_path, label = args
    try:
        # setiap worker bikin OCR sendiri
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='id',
            det_db_unclip_ratio=2,
            show_log=False,
            use_gpu=True
        )

        img = preprocess_image(image_path)
        result = ocr.ocr(img, cls=True)
        if not result or result[0] is None:
            return (os.path.basename(image_path), label, "")
        text = "\n".join([line[1][0] for line in result[0]])
        return (os.path.basename(image_path), label, text)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return (os.path.basename(image_path), label, "")
    finally:
        # bersihkan memori
        try:
            del img, result, ocr
        except:
            pass
        gc.collect()
        try:
            paddle.device.cuda.empty_cache()
        except:
            pass


def get_all_image_paths(root_folder):
    image_paths = []
    for root, dirs, files in os.walk(root_folder):
        label = os.path.basename(root)
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append((os.path.join(root, filename), label))
    return image_paths


def process_folder_multiprocessing(root_folder, output_csv, num_workers=None):
    image_paths = get_all_image_paths(root_folder)
    print(f"Processing {len(image_paths)} images with multiprocessing...")

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label', 'text'])

        with mp.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap(worker_extract, image_paths), total=len(image_paths), desc="Processing Images"):
                writer.writerow(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PaddleOCR on a folder with multiprocessing.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path ke folder dataset input")
    parser.add_argument('--output_csv', type=str, required=True, help="Path output CSV")
    parser.add_argument('--num_workers', type=int, default=1, help="Jumlah worker multiprocessing")

    args = parser.parse_args()

    process_folder_multiprocessing(args.input_folder, args.output_csv, args.num_workers)
