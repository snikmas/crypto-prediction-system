import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
import matplotlib.pyplot as plt
import joblib
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 1. Load data
def load_processed_data():

    proj_root = Path(__file__).resolve().parents[2]
    processed_dir_path = proj_root / "data" / "processed"

    if not processed_dir_path.exists():
        raise FileNotFoundError(f"Directory Not Found {[processed_dir_path]}")

    files = [f for f in processed_dir_path.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError("No parquet files found in processed")
    latest_file = max(files, key=lambda f: f.stat().st_ctime)

    logger.info(f"The last created file: {latest_file}")

    df = pd.read_parquet(latest_file, engine='pyarrow')

    return df

# 2. Split data chronologically
def split_data(df, train_ratio=0.6, val_ratio=0.2):

    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    n = len(df)

    logger.info(f"target_vol_regime hours: {len(df[df['target_vol_regime'] == 1])}")
    logger.info(f"normal hours (no target_vol_regime): {len(df[df['target_vol_regime'] == 0])}")
    logger.info(f"spike hours: {len(df[df['target_spike'] == 1])}")
    logger.info(f"normal hours (no spike): {len(df[df['target_spike'] == 0])}")

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[0:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:n]

    return train_df, val_df, test_df

# 3. Prepare features and targets
def prepare_features_targets(df, target_col):

    x_columns = df.columns.drop(["symbol", "timestamp", "created_at", "target_spike", "target_vol_regime"])
    X = df[x_columns]
    y = df[target_col]
    return X, y


# 4. Train model
def train_model(X_train, y_train):

    from xgboost import XGBClassifier

    model = XGBClassifier(
        random_state=0,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    )

    model.fit(X_train, y_train)

    return model

# 5. Evaluate model
def evaluate_model(model, X, y, threshold=0.5, dataset_name="validation"):

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    logger.info(f"Classification Report ({dataset_name}):\n{classification_report(y, y_pred)}")
    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion matrix ({dataset_name}):\n{cm}")
    logger.info(
        f"Accuracy: {accuracy_score(y, y_pred):.4f} | "
        f"Recall: {recall_score(y, y_pred):.4f} | "
        f"Precision: {precision_score(y, y_pred):.4f} | "
        f"F1: {f1_score(y, y_pred):.4f}"
    )

    return y_pred



def plot_threshold_tradeof(model, X_val, y_val, target):

    thresholds = np.arange(0.1, 0.9, 0.05)

    y_proba = model.predict_proba(X_val)[:, 1]

    recalls = []
    precisions = []
    f1s = []

    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)

        recalls.append(recall_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))

    plt.figure(figsize=(10, 6))

    plt.plot(thresholds, recalls, label="Recall", marker='o', color='blue')
    plt.plot(thresholds, precisions, label='Precision', marker='o', color='red')
    plt.plot(thresholds, f1s, label='F1', marker='o', color='green')

    plt.axvline(x=0.3, color='red', linestyle='--', label='Selected (0.3)')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold vs Metrics - {target}')
    plt.legend()
    plt.grid(True)

    plt.savefig('threshold_analysis.png')
    plt.show()



if __name__ == "__main__":
    # Load data
    df = load_processed_data()

    logger.info(f"before dropping NaN: {len(df)}")
    features = [col for col in df.columns if col not in ['symbol', 'timestamp', 'created_at', 'target_spike', 'target_vol_regime']]
    df = df.dropna(subset=features)
    logger.info(f"after dropping NaN: {len(df)}")

    pd.set_option('display.max_columns', None)

    train_df, val_df, test_df = split_data(df)

    # Train model 1: Volatility regime
    X_train, y_train = prepare_features_targets(train_df, 'target_vol_regime')
    X_val, y_val = prepare_features_targets(val_df, 'target_vol_regime')

    model_vol = train_model(X_train, y_train)
    logger.info("Results for Volatility:")
    evaluate_model(model_vol, X_val, y_val, threshold=0.5, dataset_name="validation")


    plot_threshold_tradeof(model_vol, X_val, y_val, "target_vol_regime")


    # Train model 2: Spike detector
    X_train, y_train = prepare_features_targets(train_df, "target_spike")
    X_val, y_val = prepare_features_targets(val_df, "target_spike")

    logger.info("Results for spikes:")
    model_spike = train_model(X_train, y_train)
    evaluate_model(model_spike, X_val, y_val, threshold=0.3, dataset_name="validation")


    plot_threshold_tradeof(model_spike, X_val, y_val, "target_spike")

    logger.info('=' * 50)
    logger.info("Final Test Set Evaluation")
    logger.info('=' * 50)

    X_test_vol, y_test_vol = prepare_features_targets(test_df, 'target_vol_regime')
    logger.info('Volatility Model (Test Set, Threshold 0.5)')
    evaluate_model(model_vol, X_test_vol, y_test_vol, threshold=0.5, dataset_name="test")

    logger.info("Spike Model (Test Set, Threshold 0.3)")
    X_test_spike, y_test_spike = prepare_features_targets(test_df, 'target_spike')
    evaluate_model(model_spike, X_test_spike, y_test_spike, threshold=0.3, dataset_name="test")


    models_dir = Path(__file__).resolve().parents[2] / 'models'
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model_vol, models_dir / 'volatility_xgb.pkl')
    joblib.dump(model_spike, models_dir / 'spike_xgb.pkl')

    config = {
        'volatility_threshold': 0.5,
        'spike_threshold': 0.3,
        'features': X_train.columns.tolist(),
        'model_type': 'XGBoost',
        'training_date': str(pd.Timestamp.now())
    }

    with open(models_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f'Models saved to {models_dir}')
