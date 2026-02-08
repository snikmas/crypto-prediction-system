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
    forestTree = RandomForestClassifier(random_state=0, class_weight="balanced")
    # now need regression type: 1 or 0
    forestTree.fit(X_train, y_train)

    return forestTree

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
    threshold = 0.3
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
    # plt.figure(figsize=(10,10))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.title("Confusion Matrix")
    # plt.show()

    return predictions


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
    # print(f"x train: {X_train}")
    model_vol = train_model(X_train, y_train)
    print("Results for Volatility:")
    evaluate_model(model_vol, X_val, y_val, "validation")

    # After training volatility model:
    importances = model_vol.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(importance_df.head(10))
    # impossible predict
    # Train model 2: Spike detector
    # (same process, different target)
    X_train, y_train = prepare_features_targets(train_df, "target_spike")
    X_val, y_val = prepare_features_targets(val_df, "target_spike")
    
    print("Results for spikes:")
    model_vol = train_model(X_train, y_train)
    evaluate_model(model_vol, X_val, y_val)


    # first issue: cuz we only have a few spikes, this ml is lazy: it just tells us that there're no spikes. and cuz there're only a few spikes, it will be true for 90%
    # but he doesn't predict. just chose the easiest path. so its always saying NO SPIKE and cuz there're 90/100 cases no spike - it will be ture. but he wont predict reall spikes

    # out ml model for 


    # our data is Inbalanced: there're only a few neagive SPIKES. but now ml use more accuracy parameter: correct predict / total. but FN is much worse
    # here we should use  recall: false negatives are costly 
    # recall: TP / TP + FN. and for not its.. 0%
    # so frisly have to see class distribtuion

    # solving: check how many spike hours, how many normal ohurs