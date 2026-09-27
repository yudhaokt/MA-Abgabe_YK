# MA-Abgabe_YK

Repository for the Master's thesis workflow on **Classification of Road Conditions Using Deep Learning and Explainable AI on Street View Imagery**.

**Results** folder includes several precomputed files and result of the experiment.

---

# 1. Environment Setup

## 1.1 Recommended Python Version

Create a Python virtual environment. Python **3.11.4** is recommended.


## 1.2 Install PyTorch

PyTorch should be installed separately because the installation depends on OS, GPU, CUDA version.
Use the official PyTorch installation page:

https://pytorch.org/get-started/locally/


## 1.3 Install the Remaining Requirements

After installing PyTorch, install the remaining packages:

```bash
pip install -r requirements.txt
```
---

# 2. Dataset Preparation

## 2.1 Road Quality Dataset (RQD)

The RQD dataset is not included and can be downloaded from:

https://github.com/lenoch0d/road-quality-classification

under the **Download** section in the README.

Download and extract:

```text
RQD_train.zip
RQD_test.zip
```

After extraction, the dataset should be available under **Datasets/RQD/** 
with the following structure:

```text
Datasets/RQD/
├── train/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   └── 6/
│
├── val/
│   ├── 1/
│   ├── .../
│   └── 6/
│
└── test/
    ├── 1/
    ├── .../
    └── 6/
```

The original six RQD classes are used for the first model-development experiments.


## 2.2 StreetSurfaceVis

StreetSurfaceVis is not included and can be downloaded from:

https://zenodo.org/records/11449977

Download:

```text
s_original.zip
```

and extract it into the corresponding dataset directory under:

```text
Datasets/StreetSurfaceVis/
```

Before running the StreetSurfaceVis notebook, verify the dataset path in the configuration section of the notebook.

---

# 3. Model Development

The following notebooks reproduce the model-development experiments described in the thesis.


## 3.1 MobileNet on RQD

Run:

```text
4-2_MobileNet_RQD.ipynb
```

This notebook trains and evaluates a MobileNet model using the original six-class RQD dataset.

The notebook contains the main configuration parameters near the beginning.


## 3.2 MobileNetV2 on RQD

Run:

```text
4-2_MobileNet_V2_RQD.ipynb
```

This notebook trains and evaluates MobileNetV2 using RQD.


## 3.3 YOLO11x on RQD

Run:

```text
4-2_YOLO11x_RQD.ipynb
```

This notebook contains the YOLO11x classification experiments using RQD.

### Training mode

The notebook supports two training strategies.

For **two-phase** training:

```python
TWO_PHASE = True
```

This performs:

1. an initial phase with frozen model layers,
2. a second phase with the model unfrozen for fine-tuning.

For **direct fine-tuning**:

```python
TWO_PHASE = False
```

### Image size

The tested input image sizes include:

```python
IMGSZ = 360
```

and:

```python
IMGSZ = 224
```

Check the configuration section before starting training.


## 3.4 YOLO11x on StreetSurfaceVis

Run:

```text
4-4_YOLO11x_SSV_FFT.ipynb
```

This notebook trains YOLO11x using the StreetSurfaceVis dataset.

Before execution, verify the StreetSurfaceVis dataset path.


## 3.5 Remap the RQD Dataset

Run:

```text
4-5-1_RQD_remapping.py
```

The script:

1. reads the existing RQD `train`, `val`, and `test` folders,
2. combines the original images internally,
3. remaps the six original RQD classes,
4. creates a new training/validation/test split,
5. writes the remapped dataset to the configured output directory.

The remapped dataset will contain:

```text
train/
val/
test/
```

with the six remapped class directories inside each split.


## 3.6 Train the Final YOLO11x Model

Run:

```text
4-5-3_YOLO11x_FINAL_remapped.ipynb
```

This notebook trains the final YOLO11x classification model using the remapped RQD dataset.

The resulting model is used in:

- road-feature classification,
- feature-level probability averaging,
- SHAP/XAI analysis.

Before execution, check:

- remapped dataset path,
- training output path,
- GPU/device setting,
- batch size,
- image size.

---

# 4. Model Application

The following workflow applies the final model to OSM road features for which smoothness information is missing.

The scripts should be executed in the order shown below.


## 4.1 Check Google Street View Availability

Put Google Map API Key into:

```text
src/gmap_api_key.txt
```

Run:

```text
5-1-2_Find_road_no_GSV.py
```

This script checks whether suitable Google Street View panoramas are available for the selected OSM road features.

The script creates an output dataset containing road features for which Street View imagery is available.

Before running the script, verify the configuration section, especially:

- input GeoPackage path,
- output GeoPackage path,
- Google API key,
- search radius and other GSV parameters.


## 4.2 Download Google Street View Images

Run:

```text
5-1-3_GSV_downloader.py
```

The script retrieves Google Street View images for the selected road features.

Images are grouped by OSM feature ID (`osmid`).

Example:

```text
GSV_Downloader/
└── .../
    ├── 1003267613/
    │   ├── image_001_...
    │   ├── image_002_...
    │   └── image_003_...
    │
    ├── 1003267614/
    │   ├── image_001_...
    │   └── image_002_...
    │
    └── ...
```

Several images can therefore belong to the same road feature.

The thesis workflow used approximately the following GSV retrieval settings:

```text
Point spacing          = 5 m
Search radius          = 15 m
Field of view          = 45°
Pitch                  = -30°
```

The image heading is derived from the local direction of the road geometry.

Duplicate panorama IDs are skipped.

Before execution, verify:

- Google Street View API key,
- input GeoPackage path,
- image output directory,
- retrieval parameters.

### Important

Google Street View is an external service. Panorama availability can change over time.  
Therefore, images retrieved during a later reproduction attempt may not be identical to the images originally used in the thesis.


## 4.3 Road-Feature Classification

Run:

```text
5-2a_Road_feature_classifier.py
```

This script:

1. loads the final YOLO11x classification model,
2. searches the GSV output directory,
3. classifies every image,
4. stores the class probabilities for each image,
5. groups images by `osmid`,
6. calculates a feature-level prediction using probability averaging,
7. writes the results to a CSV classification log.



## 4.4 Add Predictions to the GeoPackage

Run:

```text
5-2b_Smoothness_to_gpkg.py
```

This script transfers the feature-level classification results from the CSV classification log to the corresponding road features in the GeoPackage.

The matching is performed using the OSM road-feature identifier.

The output GeoPackage can then be opened in GIS-Software for:

- visualization,
- inspection,
- mapping,
- further GIS processing.

Before execution, verify:

- input GeoPackage path,
- classification-log CSV path,
- output GeoPackage path.


## 4.5 SHAP / XAI Analysis

The final stage of the workflow applies SHAP to selected test images in order to investigate which image regions contributed to the model predictions.


### Select Images for SHAP

In the thesis, images were selected across all six classes.
To reproduce the same images used in the thesis, extract:

```text
xai_images_thesis.zip
```

using **Extract Here**.

Using this archive is recommended when the goal is to reproduce the SHAP examples and figures presented in the thesis.

If different images are desired, run:

```text
5-3a_Select_img_for_shap.py
```

instead. The script randomly selects images for the SHAP analysis.


### Run SHAP Analysis

Run:

```text
5-3b_SHAP_multiple_image.py
```

The script:

1. loads the final YOLO11x model,
2. loads the selected XAI images,
3. performs model inference,
4. prints the predicted class probabilities,
5. calculates SHAP values,
6. creates SHAP visualizations,
7. saves the resulting figures.

The SHAP workflow uses an image masker and calculates pixel/region contributions to the model prediction.

---

# 5. Optional Utility Scripts

The following scripts are not required for reproducing the main workflow.


## 5.1 Delete Empty GSV Folders

```text
EX_Delete_empty_folder.py
```

Deletes empty `osmid` folders that may remain after the Google Street View retrieval process.


## 5.2 Rename YOLO Classes

```text
EX_Rename_YOLO-class.py
```

Utility script for renaming YOLO classification directories/classes if necessary.

---

# 6. Recommended Full Execution Order

For reproducing the entire workflow from model development to SHAP analysis, use the following order:

```text
1. Create and activate Python environment
        │
        ▼
2. Install PyTorch
        │
        ▼
3. pip install -r requirements.txt
        │
        ▼
4. Extract RQD.zip
        │
        ▼
5. Download and extract StreetSurfaceVis
        │
        ▼
6. 4-2_MobileNet_RQD.ipynb
        │
        ▼
7. 4-2_MobileNet_V2_RQD.ipynb
        │
        ▼
8. 4-2_YOLO11x_RQD.ipynb
        │
        ▼
9. 4-4_YOLO11x_SSV_FFT.ipynb
        │
        ▼
10. 4-5-1_RQD_remapping.py
        │
        ▼
11. 4-5-3_YOLO11x_FINAL_remapped.ipynb
        │
        ▼
12. 5-1-2_Find_road_no_GSV.py
        │
        ▼
13. 5-1-3_GSV_downloader.py
        │
        ▼
14. 5-2a_Road_feature_classifier.py
        │
        ▼
15. 5-2b_Smoothness_to_gpkg.py
        │
        ▼
16. 5-3a_Select_img_for_shap.py
        │
        ▼
17. 5-3b_SHAP_multiple_image.py
```

