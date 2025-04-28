import numpy as np
import joblib
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_combined_model(model_name="rf_model.pkl"):
    data = np.load("processed_data.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    iso_model = IsolationForest(contamination=0.002, random_state=42)
    iso_model.fit(X_train)
    anomaly_scores_train = iso_model.decision_function(X_train)
    anomaly_scores_test = iso_model.decision_function(X_test)

    joblib.dump(iso_model, "isolation_forest_model.pkl")

    X_train_combined = np.hstack((X_train, anomaly_scores_train.reshape(-1, 1)))
    X_test_combined = np.hstack((X_test, anomaly_scores_test.reshape(-1, 1)))

    # Train Random Forest
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight={0: class_weights[0], 1: class_weights[1]}
    )
    rf_model.fit(X_train_combined, y_train)

    joblib.dump(rf_model, model_name)
    print(f"Models saved: {model_name}, isolation_forest_model.pkl")

    return {
        "message": "Combined model training complete",
        "saved_model": model_name
    }


def evaluate_combined_model():
    data = np.load("processed_data.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load models
    iso_model = joblib.load("isolation_forest_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")

    # Add anomaly score as feature
    anomaly_scores_test = iso_model.decision_function(X_test)
    X_test_combined = np.hstack((X_test, anomaly_scores_test.reshape(-1, 1)))

    # Predict and evaluate
    y_pred = rf_model.predict(X_test_combined)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    print("\n=== Combined Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics
