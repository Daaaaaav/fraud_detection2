import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def train_isolation_forest(contamination=0.002, threshold_percentile=0.2, use_smote=False):
    data = np.load('processed_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']  
    y_test = data['y_test']

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)  # Correct resampling on y_train, not y_test
    else:
        # Optionally, you can use under-sampling here for balancing
        # Separate the majority and minority classes in the training set
        normal_data = X_train[y_train == 0]
        fraud_data = X_train[y_train == 1]
        
        # Over-sample the fraud data to match the size of the normal data
        fraud_data_upsampled = resample(fraud_data, replace=True, n_samples=len(normal_data), random_state=42)
        
        # Combine the upsampled fraud data with the normal data
        X_train = np.concatenate([normal_data, fraud_data_upsampled])
        y_train = np.concatenate([np.zeros(len(normal_data)), np.ones(len(fraud_data_upsampled))])

    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    joblib.dump(model, 'isolation_forest_model.pkl')

    # Anomaly scores
    anomaly_scores = model.decision_function(X_test)

    # Plot anomaly score distribution
    plt.hist(anomaly_scores, bins=50)
    plt.title("Anomaly Score Distribution (Test Set)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Determine threshold
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    with open("threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)

    # Map predictions
    y_pred_mapped = np.where(anomaly_scores < threshold, 1, 0)

    # Evaluate
    total = len(y_pred_mapped)
    anomaly_count = np.sum(y_pred_mapped == 1)

    stats = {
        'total_samples': total,
        'anomalies_detected': int(anomaly_count),
        'normal_detected': int(total - anomaly_count),
        'anomaly_rate_percent': round((anomaly_count / total) * 100, 4),
        'threshold_percentile': threshold_percentile,
        'threshold_value': round(float(threshold), 6),
        'accuracy': accuracy_score(y_test, y_pred_mapped),
        'precision': precision_score(y_test, y_pred_mapped, zero_division=0),
        'recall': recall_score(y_test, y_pred_mapped, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_mapped, zero_division=0)
    }

    print("\n=== Isolation Forest Training Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def detect_anomalies():
    # Load data
    df = pd.read_csv("creditcard.csv")
    model = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]

    # Preprocess
    X = df.drop(columns=["Class"])
    y_true = df["Class"]
    X_scaled = scaler.transform(X)

    # Predict
    scores = model.decision_function(X_scaled)
    preds_mapped = np.where(scores < threshold, 1, 0)

    df["predicted"] = preds_mapped
    df["is_fraud_predicted"] = df["predicted"] == 1
    df["Actual_Label"] = df["Class"].map({0: "Not Fraudulent", 1: "Fraudulent"})
    df["Anomaly_Label"] = df["predicted"].map({0: "Normal", 1: "Anomaly"})

    # Evaluate
    true_frauds = df[(df["Class"] == 1) & (df["predicted"] == 1)]
    missed_frauds = df[(df["Class"] == 1) & (df["predicted"] == 0)]
    normal_data = df[df["Class"] == 0]

    print(f"Detected {len(true_frauds)} true frauds.")
    print(f"Missed {len(missed_frauds)} frauds.")

    # Smart sampling
    true_frauds_sample = true_frauds.sample(n=min(len(true_frauds), 50), random_state=42)

    if len(true_frauds_sample) < 2 and len(missed_frauds) > 0:
        # If very few true frauds detected, add missed frauds to sample
        needed = 2 - len(true_frauds_sample)
        extra_frauds = missed_frauds.sample(n=min(needed, len(missed_frauds)), random_state=42)
        true_frauds_sample = pd.concat([true_frauds_sample, extra_frauds])

    remaining = 100 - len(true_frauds_sample)
    other_data_sample = normal_data.sample(n=remaining, random_state=42)

    final_sample = pd.concat([true_frauds_sample, other_data_sample])
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    # Visualization (Optional: Anomaly score vs Actual label)
    plt.figure(figsize=(10, 6))
    plt.hist(scores[y_true == 0], bins=50, alpha=0.6, label="Normal")
    plt.hist(scores[y_true == 1], bins=50, alpha=0.6, label="Fraud")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")
    plt.title("Anomaly Scores - Normal vs Fraud")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    stats = {
        "threshold_used": float(threshold),
        "accuracy": accuracy_score(y_true, df["predicted"]),
        "precision": precision_score(y_true, df["predicted"], zero_division=0),
        "recall": recall_score(y_true, df["predicted"], zero_division=0),
        "f1_score": f1_score(y_true, df["predicted"], zero_division=0),
        "true_positives_detected": int(len(true_frauds)),
        "missed_frauds": int(len(missed_frauds)),
    }

    print("\n=== Detection Stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return {
        "evaluated_sample": final_sample.to_dict(orient="records"),
        "stats": stats
    }
