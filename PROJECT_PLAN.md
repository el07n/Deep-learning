# SmartPlate AI - execution and rubric plan

## Scope

Build one Streamlit Computer Vision application around an ImageNet-pretrained
EfficientNetV2B0. The adapted model recognizes frequent food components and
directly regresses calories and protein from an uploaded RGB meal image.

## Milestones

1. Acquire the Nutrition5k metadata, official splits, and overhead RGB subset.
2. Prepare a dish-level manifest, cleaning report, statistics, and EDA plots.
3. Train new multi-task heads on a frozen pre-trained backbone.
4. Fine-tune the final backbone blocks with a low learning rate.
5. Evaluate recognition and both regression targets on the held-out test set.
6. Calibrate nutrition prediction intervals on the validation split.
7. Integrate the exported artifacts into Streamlit and test the full flow.
8. Create the slide deck and rehearse the live demonstration.

## Definition of done

- Source code and dependency file run from a clean Python environment.
- Dataset statistics include counts, distributions, and cleaning exclusions.
- No dish ID appears in more than one split.
- Model artifacts record the vocabulary, target scaling, and configuration.
- Test-set metrics are compared with a mean-prediction baseline.
- Streamlit includes title, description, upload, prediction button, result, and
  a confidence score, with a visible limitation disclaimer.
- Slides contain every required rubric section, team names, and student IDs.

