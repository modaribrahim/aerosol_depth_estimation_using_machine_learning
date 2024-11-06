# Solafune AOD Estimation Challenge Solution

This repository contains the solution to the Solafune Aerosol Optical Depth (AOD) Estimation challenge.
In this project,AOD values are estimated from Sentinel-2 Level-2A multispectral images by extracting and analyzing spectral and texture-based features.
The final model, a CatBoost regressor, achieved a Pearson correlation coefficient of nearly 0.98 on the private leaderboard.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Feature Extraction Process](#feature-extraction-process)
- [Data Processing and Pipeline](#data-processing-and-pipeline)
- [Training the Model](#training-the-model)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

## Project Overview

### Objective

The aim of this project is to accurately estimate Aerosol Optical Depth (AOD) values from 13-band Sentinel-2 imagery.
This estimation is crucial for understanding air quality and climate-related studies.

### Methodology

- **Feature Engineering**: Extracted statistical, spectral, and texture-based features from Sentinel-2 images.
- **Feature Types**:
  - **Basic statistics**: Mean, variance, median, skewness, kurtosis, percentiles, etc.
  - **Spectral indices**: Calculated using band combinations to capture vegetation health, bloom, red edge characteristics, and more.
  - **Texture features**: Extracted using GLCM (Gray-Level Co-occurrence Matrix) properties such as contrast, correlation, and homogeneity.
- **Model**: The features were used to train a CatBoost regressor that achieved high accuracy on the private leaderboard.

## Installation

### Dependencies

The repository requires Python 3.8+ and the following libraries:

- `numpy`
- `pandas`
- `rasterio`
- `scipy`
- `skimage`
- `tqdm`
- `cv2` (OpenCV)
- `joblib`
- `scikit-learn`
- `CatBoost`

Install the required libraries using:

```bash
pip install -r requirements.txt
```
## Additional Data

Due to contest restrictions, the raw data is not publicly available. Ensure you have access to the private dataset files, as provided by the contest organizers.

## Feature Extraction Process

### Script Overview

The main feature extraction pipeline is implemented in `feature_extraction.py` and `feature_extraction_utilites.py` and includes:

- **Basic Statistical Features**: Mean, median, variance, kurtosis, skewness, and percentiles (10th, 25th, 75th, 90th).
- **Band Ratios**: Calculated using different band combinations to capture unique spectral information.
- **Advanced Band Features**: Includes geometric mean, harmonic mean, normalized ratio, and custom band transformations.
- **Texture Features**: Extracted GLCM features (contrast, correlation, energy, homogeneity) to capture spatial patterns.

### Functions

- `calculate_statistics`: Computes basic statistics for each band.
- `calculate_advanced_band_features`: Derives more complex transformations between bands.
- `compute_texture_features`: Extracts texture features using GLCM properties.
- `get_usable_bands`: Filters out bands with low data usability (e.g., high proportion of missing values).

## Data Processing and Pipeline

The script `process_data` in `feature_extraction.py` orchestrates the data processing pipeline:

1. **Load Train and Test Data**: Reads metadata files for both train and test sets.
2. **Process Bands**: Loads the band data for each image and reshapes it into a usable format.
3. **Compute Indices and Features**: Calls various functions to calculate vegetation indices, red edge indices, statistical features, and texture features.
4. **Normalize Features**: Uses `QuantileTransformer` and `KNNImputer` to handle missing values and normalize feature distributions.

## Training the Model

The CatBoost model is trained on extracted features saved in `train_features_root.npy` and `train_labels_root.npy`. These files contain processed feature arrays and corresponding AOD values.

### Model Training

To train the model, use the following command:

```bash
python train_model.py --features_path /path/to/features --labels_path /path/to/labels
```

### Running Feature Extraction

To run the feature extraction pipeline, specify paths to input files and directories:

```bash
python feature_extraction.py --train_csv train_data.csv --test_csv test_data.csv --output_dir ./output --bands_dir /path/to/bands
```
## File Structure
- `scripts`: Contains python files.
- `notebooks`: Contains jupyter notebooks.
- `feature_extraction.py`: Main script for extracting features.
- `train_model.py`: Script to train the CatBoost model on extracted features.
- `optimize_parameters.py`: Script to optimize model paramteres using optuna.
- `data_processing.py`: Loads features to be fed to catboost model.
- `feature_extraction_utilites.py`: Contains utilites for features extraction pipeline.

## Acknowledgments

This project is built upon Sentinel-2 data provided by the contest organizers.

For any inquiries or further collaboration, please reach out.
