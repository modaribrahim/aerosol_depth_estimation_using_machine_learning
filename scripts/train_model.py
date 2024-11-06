# train_model.py
import argparse
import logging
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr
from data_processing import load_data

logging.basicConfig(level=logging.INFO, filename="train_catboost.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def train_catboost(X_data, y_data, params, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y_data))
    models = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_data)):
        logging.info(f"Training fold {fold + 1}/{n_splits}")
        X_train, X_valid = X_data[train_idx], X_data[valid_idx]
        y_train, y_valid = y_data[train_idx], y_data[valid_idx]
        
        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)
        model = CatBoostRegressor(**params)
        
        model.fit(train_pool, eval_set=valid_pool, verbose=100)
        models.append(model)
        oof_predictions[valid_idx] = model.predict(X_valid)

        pearson_corr = pearsonr(y_valid, oof_predictions[valid_idx])[0]
        logging.info(f"Fold {fold + 1} Pearson R: {pearson_corr:.4f}")
    
    overall_pearson_corr = pearsonr(y_data, oof_predictions)[0]
    logging.info(f"Overall Pearson R: {overall_pearson_corr:.4f}")
    
    return models, oof_predictions

def main(args):
    train_features, test_features, train_targets, train_groups, test_df = load_data()

    catboost_params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg
    }

    models, oof_predictions = train_catboost(train_features, train_targets, catboost_params)
    np.save(args.output_oof, oof_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--output_oof", type=str, default="catboost_oof_predictions.npy")
    args = parser.parse_args()
    
    main(args)
