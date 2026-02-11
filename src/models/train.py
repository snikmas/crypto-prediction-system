import pandas as pd
import numpy as np
from pathlib import Path
# from sklearn.model_selection import train_test_split  # Actually, DON'T use this (time series)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json


# 1. Load data
def load_processed_data():
    """
    TODO:
    - Load features.parquet (find latest if multiple files)
    - Return DataFrame
    """
    proj_root = Path(__file__).resolve().parents[2]
    # do we need to check does it exist?
    processed_dir_path = proj_root / "data" / "processed"
    if not processed_dir_path.exists():
        raise FileNotFoundError(f"Directory Not Found {[processed_dir_path]}")

    files = [f for f in processed_dir_path.iterdir() if f.is_file()] # check only files
    if not files:
        raise FileNotFoundError("No parquet files found in processed")
    latest_file = max(files, key=lambda f: f.stat().st_ctime)

    logging.info(f"The last created file: {latest_file}")

    
    df = pd.read_parquet(latest_file, engine='pyarrow')
    
    
    return df

# 2. Split data chronologically
def split_data(df, train_ratio=0.6, val_ratio=0.2):
    """
    TODO:
    - Sort by timestamp (should already be sorted, but verify)
    - Split into train/val/test by index position (NOT random)
    - Example: 0-60% train, 60-80% val, 80-100% test
    - Return: train_df, val_df, test_df
    """
    df = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)# from oldest to newest
    
    # get N of rows
    n = len(df)


    # Look at vol hours
    spike_hours = df[df['target_vol_regime'] == 1]
    print(f"target_vol_regime hours: {len(spike_hours)}")

    # Look at normal vol hours
    normal_hours = df[df['target_vol_regime'] == 0]
    print(f"normal hours (no target_vol_regime): {len(normal_hours)}")



    # Look at spike hours
    spike_hours = df[df['target_spike'] == 1]
    print(f"spike hours: {len(spike_hours)}")

    # Look at normal hours
    normal_hours = df[df['target_spike'] == 0]
    print(f"normal hours (no spike): {len(normal_hours)}")


    # calculate it. what we do: frist N rows for training; second N rows for validation, third N orws to test
    train_end_idx = int(n * train_ratio) # 0.6 from 7000
    val_end_idx = int(n * (train_ratio + val_ratio)) # 0.8 from 7000

    train_df = df.iloc[0:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:n]
    
    return train_df, val_df, test_df

# 3. Prepare features and targets
def prepare_features_targets(df, target_col):
    """
    TODO:
    - Drop rows with NaN in ANY feature column
    - Separate X (features) and y (target)
    - Features to EXCLUDE: symbol, timestamp, created_at, both targets
    - Return: X, y
    """
    
    # not sure about target spike and why here we only calculate for target_vol_regime. but 
    x_columns = df.columns.drop(["symbol", "timestamp", "created_at", "target_spike", "target_vol_regime"])
    X = df[x_columns]
    y = df[target_col]
    return X, y
    

# 4. Train model
def train_model(X_train, y_train):
    """
    TODO:
    - Create RandomForestClassifier
    - Fit on training data
    - Return trained model
    """
    logging.info(X_train)
    logging.info(y_train)
    # forestTree = RandomForestClassifier(random_state=0, class_weight="balanced")
    # now need regression type: 1 or 0
    # forestTree.fit(X_train, y_train)
    from xgboost import XGBClassifier

    model = XGBClassifier(
        random_state=0,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
    )

    model.fit(X_train, y_train)

    return model

# 5. Evaluate model
def evaluate_model(model, X, y, dataset_name="validation"):
    """
    TODO:
    - Predict on X
    - Print classification_report (precision, recall, F1)
    - Print confusion_matrix
    - Return predictions
    """
    predictions = model.predict(X)
    logging.info(f"Classification Report:\n{classification_report(y, predictions)}")

    # TEST TO CHANGE THRESHOLD. STILL DOES NOT WORK
    y_proba = model.predict_proba(X)[:, 1]
    threshold = 0.5
    y_pred = (y_proba > threshold).astype(int)

    predictions = y_pred
    logging.info(f'\nconfusion matrix:')
    cm = confusion_matrix(y, predictions)
    print(cm)
    print(f'''
        Accuracy score:  {accuracy_score(y, predictions)}
        Recall score:    {recall_score(y, predictions)} 
        Precision score: {precision_score(y, predictions)}
        F1 score:        {f1_score(y, predictions)}
        ''')    
    
    
    return predictions



def plot_threshold_tradeof(model_vol, X_val, y_val, target):
     # 1. Create threshold array
    thresholds = []
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    # 2. Initialize empty lists
    recalls = []
    precisions = []
    f1s = []
    
    # 3. Loop through thresholds
    for thresh in thresholds:
        # Get probabilities
        y_proba = model_vol.predict_proba(X_val)[:, 1]
        
        # Apply threshold   
        y_pred = (y_proba > thresh).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Append to lists
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    
    # 4. Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(thresholds, recalls, label="Recall", marker='o', color='blue')  # Recall
    plt.plot(thresholds, precisions, label='Precicison', marker='o', color='red')  # Precision
    plt.plot(thresholds, f1s, label='F1', marker='o', color='green')  # F1

    # Add vertical line at selected threshold
    plt.axvline(x=0.3, color='red', linestyle='--', label='Selected (0.3)')
    
    # Add labels and formatting
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Metrics - Spike Detection')
    plt.legend()
    plt.grid(True)
    
    # Save
    plt.savefig('threshold_analysis.png')
    plt.show()



if __name__ == "__main__":
    # Load data
    df = load_processed_data()
    
    # dropping it here otherwise i 
    print(f"before dropping: {len(df)}")
    features = [col for col in df.columns if col not in ['symbol', 'timestamp', 'created_at', 'target_spike', 'target_vol_regime']]
    df = df.dropna(subset=features)
    print(f"after dropping: {len(df)}") # NO NEED INPLACE, JUST DROPS ROWS
    
    pd.set_option('display.max_columns', None)

    # why after dropping it it still exists?
    # Split
    train_df, val_df, test_df = split_data(df) # it not null
    
    # Train model 1: Volatility regime
    X_train, y_train = prepare_features_targets(train_df, 'target_vol_regime')
    X_val, y_val = prepare_features_targets(val_df, 'target_vol_regime')
    

    #x is empty?
    # print(f"x train: {X_train}") # for volatility XBOOT is okay 
    model_vol = train_model(X_train, y_train)
    print("Results for Volatility:")
    evaluate_model(model_vol, X_val, y_val, "validation")

    
    plot_threshold_tradeof(model_vol, X_val, y_val, "target_vol_regime")
    # Train model 2: Spike detector
    X_train, y_train = prepare_features_targets(train_df, "target_spike")
    X_val, y_val = prepare_features_targets(val_df, "target_spike")
    
    print("Results for spikes:")
    model_vol = train_model(X_train, y_train)
    evaluate_model(model_vol, X_val, y_val)


    plot_threshold_tradeof(model_vol, X_val, y_val, "target_spike")

    # final evaluation on test set
    print('\n' + '=' * 50)
    print("Final Test Set Evaluation")
    print('=' * 50)

    # volatility model
    X_test_vol, y_test_vol = prepare_features_targets(test_df, 'target_vol_regime')
    print('\n==== Volatility Model (Test Set, Threshold 0.5) ====')
    y_proba = model_vol.predict_proba(X_test_vol)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    print(confusion_matrix(y_test_vol, y_pred))
    print(f"Recall: {recall_score(y_test_vol, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test_vol, y_pred):.2%}")
    print(f"F1: {f1_score(y_test_vol, y_pred):.2%}")


    # spike model
    print("\n=== Spike Model (Test Set, Threshold 0.3) ===")
    X_test_spike, y_test_spike = prepare_features_targets(test_df, 'target_spike')
    y_proba = model_vol.predict_proba(X_test_spike)[:, 1]
    y_pred = (y_proba > 0.3).astype(int)
    print(confusion_matrix(y_test_spike, y_pred))
    print(f"Recall: {recall_score(y_test_spike, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test_spike, y_pred):.2%}")
    print(f"F1: {f1_score(y_test_spike, y_pred):.2%}")


    # saving the model
    models_dir = Path(__file__).resolve().parents[2] / 'models'
    models_dir.mkdir(exist_ok=True)

    joblib.dump(model_vol, models_dir / 'volatility_xgb.pkl')
    joblib.dump(model_vol, models_dir / 'spike_xgb.pkl')

    config = {
        'volatility_threshold': 0.5,
        'spike_threshold':0.3,
        'features': X_train.columns.tolist(),
        'model_type': 'XGBoost',
        'training_date': str(pd.Timestamp.now())
    }

    with open(models_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'\n\n✓ Models saved to {models_dir}')