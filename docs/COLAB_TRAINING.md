# Final training on Google Colab

The local machine currently exposes no TensorFlow GPU. Use a Colab GPU for the
final experiment so the submitted metrics come from ImageNet transfer learning
on the complete prepared dataset.

## 1. Put the source in Google Drive

Copy the project source (excluding `.venv`, `data/raw`, `tmp`, and `artifacts`)
to `MyDrive/SmartPlateAI`. The public images will be downloaded directly inside
the Colab runtime, and final artifacts will be saved back to Drive.

## 2. Set the Colab runtime

Choose **Runtime > Change runtime type > T4 GPU**. Then run:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/SmartPlateAI
!python -m pip install -q -r requirements.txt
```

Confirm that a GPU is visible:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

## 3. Download and prepare Nutrition5k

Runtime storage is used for the public dataset so Drive is not slowed by
thousands of small image files.

```python
DATASET = "/content/nutrition5k_dataset"
PREPARED = "/content/smartplate_processed"

!python -m scripts.download_nutrition5k_support_files --dataset-root "$DATASET"
!python -m scripts.download_nutrition5k_subset --dataset-root "$DATASET" --workers 16
!python -m scripts.prepare_nutrition5k \
    --dataset-root "$DATASET" \
    --output-dir "$PREPARED" \
    --max-ingredients 50 \
    --exclude-ids "data/exclude_dish_ids.txt"
```

Expected clean split counts are 2,321 train, 432 validation, and 506 test.
Three published overhead IDs return HTTP 404 and are recorded as unavailable.

## 4. Train and evaluate

```python
ARTIFACTS = "/content/drive/MyDrive/SmartPlateAI/artifacts"

!python -m scripts.train \
    --prepared-dir "$PREPARED" \
    --output-dir "$ARTIFACTS" \
    --head-epochs 12 \
    --fine-tune-epochs 12 \
    --batch-size 16

!python -m scripts.evaluate \
    --prepared-dir "$PREPARED" \
    --artifacts-dir "$ARTIFACTS" \
    --threshold 0.50 \
    --coverage 0.95
```

Do not add `--weights none`: the default `imagenet` weights are required by the
assignment. Early stopping may finish either phase before 12 epochs.

## 5. Collect submission results

Use these generated files in the slides:

- `artifacts/evaluation_metrics.json`;
- `artifacts/predicted_vs_actual.png`;
- `artifacts/training_log.csv`;
- `data/processed/plots/nutrition_distributions.png`;
- `data/processed/plots/ingredient_frequencies.png`.

Copy the two EDA plots from the local project or change `PREPARED` to a Drive
folder if they should be generated directly there. After the final artifacts
are present in the project root, run `streamlit run app.py` locally for the demo.
