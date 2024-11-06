import joblib
import optuna
from train_model import train_catboost
from data_processing import load_data

def objective(trial):
    catboost_params = {
        "iterations": trial.suggest_int("iterations", 1000, 1300),
        "depth": trial.suggest_int("depth", 6, 8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.07),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 10)
    }
    
    train_features, _, train_targets, _, _ = load_data()
    _, oof_predictions = train_catboost(train_features, train_targets, catboost_params)
    overall_pearson_corr = pearsonr(train_targets, oof_predictions)[0]
    
    return -overall_pearson_corr

if __name__ == "__main__":
    study_name = "catboost_study"
    study_path = "catboost_study.pkl"
    
    try:
        study = joblib.load(study_path)
        print("Loaded existing study.")
    except FileNotFoundError:
        study = optuna.create_study(direction="minimize")
        print("Created new study.")

    study.optimize(objective, n_trials=3)
    joblib.dump(study, study_path)

    print("Best hyperparameters: ", study.best_trial.params)
