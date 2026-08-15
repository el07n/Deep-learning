# Assignment requirements checklist

Reviewed against `202605 - Group Project (1).pdf` (4 pages).

## Project-level requirements

| Requirement | Implementation/evidence | Status |
|---|---|---|
| Group of 3-5 students | Add names and IDs to the final slides | Team input needed |
| Computer Vision or NLP problem | Food-image recognition and nutrition estimation | Complete |
| Do not build the core model from scratch | Keras EfficientNetV2B0 with ImageNet weights | Complete in code |
| Transfer learning/fine-tuning/inference | Frozen-backbone phase followed by low-LR fine-tuning | Complete |
| Real-world problem and target users | Meal nutrition estimation for users tracking food intake | Documented |
| Suitable public dataset | Official Google Research Nutrition5k | Verified |
| Submission: slides and source code | Source code and slide outline exist | Final PPTX pending |
| Due date | 27 August 2026 | Confirmed from brief |

## Table 2 implementation requirements

| Required item | Project location | Status |
|---|---|---|
| Problem and motivation | `PROJECT_PLAN.md`, slide outline | Complete |
| Model selection and description | `smartplate/model.py`, `README.md` | Complete |
| Dataset collection | official-file download scripts | Complete in code |
| Preprocessing | RGB decode/resize, augmentation, target scaling, label encoding | Complete |
| Cleaning | duplicates, missing/invalid data, corrupt images, manual outlier list | Complete |
| Dataset statistics | JSON statistics and two EDA plots | Complete in code |
| Model training | frozen head and fine-tuning stages | Complete on GPU |
| Model evaluation | F1/precision/recall and MAE/RMSE/R2/PMAE | Complete |
| Baseline comparison | train-mean calories/protein baseline | Complete |
| Streamlit title and description | `app.py` | Complete |
| Image upload | `st.file_uploader` | Complete |
| Prediction button | explicit Predict button | Complete |
| Prediction result | ingredients, calories, protein | Complete |
| Confidence score | sigmoid confidence per recognized component | Complete |
| Nutrition uncertainty | validation-residual empirical ranges | Complete |
| Advantages and limitations | `README.md`, Streamlit About tab | Complete |
| Live demonstration | slide outline includes a demo plan | Rehearsal pending |

## Verification already completed

- Parsed the two real official metadata files: 5,006 unique dish records.
- Confirmed the official overhead split: 2,758 train IDs and 507 test IDs.
- Downloaded 3,262 available official overhead RGB images (1.22 GB). Three IDs
  published in the split return HTTP 404 from the official storage bucket.
- Prepared 3,259 clean dishes after removing two invalid nutrition records and
  one visually confirmed metadata outlier.
- Verified 2,321/432/506 train-validation-test samples, 3,259 unique dish IDs,
  no split overlap, no missing manifest paths, and a 50-label training-only
  vocabulary covering 85.0% of training dishes.
- Ran preparation, frozen training, fine-tuning, evaluation, artifact loading,
  single-image prediction, and a Streamlit health check end to end.
- Completed final ImageNet transfer learning and fine-tuning on GPU.
- Evaluated once on the 506-dish official test set. Calories MAE is 63.91 kcal
  with R2 0.821; protein MAE is 6.72 g with R2 0.704; recognition micro F1 is
  0.567.

## Work still required for final submission

1. Create the final PowerPoint deck using the prepared outline and results.
2. Add group members and IDs.
3. Rehearse the live Streamlit demonstration.
