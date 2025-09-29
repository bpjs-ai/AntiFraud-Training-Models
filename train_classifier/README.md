Hybrid OCR Training Pipeline

Cara pakai cepat:

1) Siapkan CSV hasil OCR yang berisi kolom: `filename`, `label`, `text` (contoh: `ocr_results_paddle.csv`). Struktur folder gambar: `Augmented/<label>/<filename>`.

2) Jalankan training default:

```
python -m train_pipeline.train
```

3) Override via config JSON (opsional):

```
python -m train_pipeline.train --config config.json
```

Contoh `config.json` minimal:

```
{
  "data_csv_path": "ocr_results_paddle.csv",
  "image_root_dir": "Augmented",
  "epochs": 50,
  "batch_size": 32,
  "lr_bert": 2e-5,
  "lr_resnet": 1e-4,
  "lr_heads": 1e-4
}
```

4) Override cepat via CLI:

```
python -m train_pipeline.train --epochs 100 --batch_size 16 --val_size 0.1
```

Output disimpan ke folder `outputs`: `best_model.pt`, `best_model_quantized.pt`, dan `label2id.json`.


