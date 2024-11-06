import argparse
import logging
import os
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
import joblib
from feature_extraction_utilites import rededge_indices, calculate_statistics, compute_texture_features, calculate_vegetation_indices, calculate_vegetation_indices2, calculate_advanced_band_features
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(train_csv, test_csv, output_dir, bands_dir):
    logging.info("Loading train data...")
    train_df = pd.read_csv(train_csv, header=None)
    train_df.columns = ["file", "location", "aod"]
    train_df["path"] = train_df["file"].apply(lambda x: os.path.join(bands_dir, "train_images", str(x)))
    
    logging.info("Loading test data...")
    test_df = pd.read_csv(test_csv, header=None)
    test_df.columns = ["file", "aod"]
    test_df["path"] = test_df["file"].apply(lambda x: os.path.join(bands_dir, "test_images", str(x)))

    logging.info(f'Train data shape: {train_df.shape}')
    logging.info(f'Test data shape: {test_df.shape}')
    
    aod = test_df['aod'].values
    bands = []

    logging.info("Processing bands...")
    for i in tqdm(range(len(train_df))):
        path = train_df.path.values[i]
        pixels = rasterio.open(path).read()
        pixels = np.reshape(pixels, (13, 128*128))
        pixels[pixels < 0] = 0
        bands.append(pixels)

    bands = np.array(bands)
    logging.info(f"Bands shape: {bands.shape}")

    logging.info("Handling usable bands...")
    usable, usable_idx = get_usable_bands(bands)

    np.save(os.path.join(output_dir, 'root_usable.npy'), usable)
    np.save(os.path.join(output_dir, 'root_usable_idx.npy'), usable_idx)

    logging.info("Computing red edge indices...")
    train_df = rededge_indices(train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'red_edge.csv'))

    logging.info("Computing additional features...")
    train_df = calculate_statistics(basic_bands, train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'stats.csv'))

    train_df = bloom_indices(train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'bloom.csv'))

    train_df = compute_texture_features(basic_bands, train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'texture.csv'))

    train_df = calculate_vegetation_indices(train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'vegetation.csv'))

    train_df = calculate_vegetation_indices2(train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'vegetation2.csv'))

    train_df = calculate_advanced_band_features(basic_bands, train_df, usable)
    train_df.to_csv(os.path.join(output_dir, 'stats_heavy.csv'))

    logging.info("Normalizing features...")
    features, targets = normalize_features(train_df, aod)
    
    np.save(os.path.join(output_dir, 'train_features_root.npy'), features)
    np.save(os.path.join(output_dir, 'train_labels_root.npy'), targets)
    train_df.to_csv(os.path.join(output_dir, 'train_df.csv'))

    return features, targets


def get_usable_bands(bands):
    usable = []
    usable_idx = []
    for idx, band in enumerate(bands):
        percent_positive = np.min(np.sum(band > 0, axis=1) / 16384)
        if 0.90 <= percent_positive <= 1:
            usable.append(band)
            usable_idx.append(idx)

    usable = np.array(usable)
    usable_idx = np.array(usable_idx)
    return usable, usable_idx


def normalize_features(train_df, aod):
    logging.info("Normalizing features...")
    scaler = QuantileTransformer(output_distribution='normal')
    train_df_scaling = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    
    logging.info("Imputing missing values...")
    for col in tqdm(train_df_scaling.columns.values):
        col_vals = train_df_scaling[col].replace([np.inf, -np.inf], np.nan)
        col_mean = np.nanmean(col_vals)
        train_df_scaling[col].replace([np.inf, -np.inf, np.nan], col_mean, inplace=True)

    train_df_scaling = train_df_scaling.loc[:, train_df_scaling.std() > 1e-5]
    
    features = np.array(train_df_scaling)
    targets = np.array(aod)

    return features, targets


def main():
    parser = argparse.ArgumentParser(description="Preprocess satellite image data and compute features.")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save preprocessed data.")
    parser.add_argument('--bands_dir', type=str, required=True, help="Directory containing image bands.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    features, targets = process_data(args.train_csv, args.test_csv, args.output_dir, args.bands_dir)

    logging.info(f"Preprocessing complete. Features and labels saved in {args.output_dir}.")


if __name__ == '__main__':
    main()
