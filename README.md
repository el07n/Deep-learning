# SmartPlate AI

SmartPlate is a Computer Vision project that uses an ImageNet-pretrained
EfficientNetV2B0 as its main component. A shared visual backbone feeds two task
heads:

- multi-label recognition of common food components;
- direct regression of total calories (kcal) and protein (g).

The prediction path is local. It does not call a nutrition API or retrieve the
answer from a food database.

## One-click Google Colab notebooks

| Notebook | Open in Colab |
|---|---|
| Nutrition5k download, cleaning, preprocessing, statistics, and plots | [![Open Data Preparation in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/el07n/Deep-learning/blob/main/notebooks/01_data_preparation.ipynb) |
| EfficientNetV2B0 model, transfer learning, fine-tuning, and evaluation | [![Open Model Training in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/el07n/Deep-learning/blob/main/notebooks/02_model_training.ipynb) |

The model notebook is standalone: if prepared data is not already available in
the active Colab runtime, it downloads and prepares the official overhead RGB
subset before training. Select a T4 GPU before running the model notebook.

## Why this satisfies the assignment

- **Existing pre-trained model:** Keras EfficientNetV2B0 with ImageNet weights.
- **Transfer learning:** the frozen backbone is trained with new task heads, then
  its last blocks are fine-tuned at a low learning rate.
- **Public dataset:** Google Research Nutrition5k provides RGB imagery,
  ingredient labels, calories, protein, official dish-level splits, and an
  evaluation protocol.
- **Preprocessing and cleaning:** image decoding checks, missing-value checks,
  non-negative nutrition validation, duplicate removal, augmentation, target
  scaling, and dish-level splitting.
- **Evaluation:** multi-label F1/precision/recall and nutrition MAE/RMSE/R2/PMAE,
  plus a mean-prediction baseline and predicted-vs-actual visualization.
- **Streamlit:** title, description, upload, prediction button, results,
  recognition confidence, nutrition uncertainty ranges, and limitations.

## Trusted sources

- [Keras EfficientNetV2B0](https://keras.io/api/applications/efficientnet_v2/efficientnet_v2_models/)
- [Keras transfer learning guide](https://keras.io/guides/transfer_learning/)
- [Official Nutrition5k repository](https://github.com/google-research-datasets/Nutrition5k)
- [Nutrition5k CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.html)

## Expected dataset layout

Do not download the full 181 GB archive unless it is genuinely needed. The
current project uses metadata, official split files, and overhead RGB images:

```text
data/raw/nutrition5k_dataset/
├── metadata/
│   ├── dish_metadata_cafe1.csv
│   └── dish_metadata_cafe2.csv
├── dish_ids/splits/
│   └── ... official train/test id files ...
└── imagery/realsense_overhead/
    └── dish_XXXXXXXXXX/
        ├── rgb.png
        └── ... depth files are ignored ...
```

Download the small official metadata and overhead split files first:

```powershell
python -m scripts.download_nutrition5k_support_files `
  --dataset-root "data/raw/nutrition5k_dataset"
```

The downloader uses the public Google Cloud Storage URLs documented by the
official repository. The image command below downloads only `rgb.png`, not the
depth files or side-angle videos.

## Environment setup

Python 3.10-3.12 is recommended. Training is much faster in Google Colab with a
GPU, but the exported model can be used locally by Streamlit.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Prepare and inspect the data

Download only the RGB image from the official overhead split. This avoids the
raw depth images and the much larger side-angle videos:

```powershell
python -m scripts.download_nutrition5k_subset `
  --dataset-root "data/raw/nutrition5k_dataset" `
  --workers 8
```

For a quick software smoke test, add `--max-train 20 --max-test 5`. Do not use
that tiny sample for final model results.

```powershell
python -m scripts.prepare_nutrition5k `
  --dataset-root "data/raw/nutrition5k_dataset" `
  --output-dir "data/processed" `
  --max-ingredients 50 `
  --exclude-ids "data/exclude_dish_ids.txt"
```

Copy `data/exclude_dish_ids.example.txt` to `data/exclude_dish_ids.txt` and add
IDs rejected during manual visual inspection. Generated outputs include:

- `manifest.csv`;
- `ingredient_vocabulary.json`;
- `dataset_statistics.json`;
- nutrition-distribution and ingredient-frequency plots.
- `manual_review_candidates.csv` for high-energy records that require visual
  inspection before being added to the exclusion list.

The final 50-label vocabulary is learned from the training split only. On the
current clean release it covers 85.0% of training dishes, and its least frequent
included label still has 113 training examples.

If official split files are present, their test set is preserved. Otherwise the
script records that fact and creates a deterministic dish-level test split.

## 2. Train and fine-tune

```powershell
python -m scripts.train `
  --prepared-dir "data/processed" `
  --output-dir "artifacts" `
  --head-epochs 12 `
  --fine-tune-epochs 12
```

The first run downloads official ImageNet weights. Use `--weights none` only for
software testing; that option does **not** satisfy the final assignment.

## 3. Evaluate and calibrate uncertainty

```powershell
python -m scripts.evaluate `
  --prepared-dir "data/processed" `
  --artifacts-dir "artifacts" `
  --threshold 0.50 `
  --coverage 0.95
```

The command writes final metrics, test predictions, uncertainty calibration, and
a predicted-vs-actual figure. Nutrition uncertainty is an empirical interval
calibrated on validation residuals; it is not presented as a fake softmax score.

## 4. Run Streamlit

```powershell
streamlit run app.py
```

The repository includes the final trained artifacts, so the app is ready for
local inference after the dependencies are installed. It never returns
placeholder predictions.

## Final test-set results

| Task | Metric | Result |
|---|---:|---:|
| Ingredient recognition | Micro F1 | 0.567 |
| Ingredient recognition | Macro F1 | 0.443 |
| Calories | MAE | 63.91 kcal |
| Calories | R2 | 0.821 |
| Protein | MAE | 6.72 g |
| Protein | R2 | 0.704 |

The held-out official test split contains 506 dishes. Full metrics, test
predictions, training logs, and the predicted-vs-actual figure are stored in
`artifacts/`.

## Important limitations

- A single RGB image does not provide an exact portion volume.
- Nutrition5k is biased toward a small number of US cafeteria locations.
- Recognition is limited to the selected frequent ingredient labels.
- Results are research estimates and must not be presented as medical advice.
