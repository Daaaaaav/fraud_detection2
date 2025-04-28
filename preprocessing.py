import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTETomek
import joblib

def preprocess_data():
    try:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, 'creditcard.csv')
        df = pd.read_csv(file_path)

        X = df.drop('Class', axis=1)
        y = df['Class']

        # Train-test split with stratification based on target class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Original class distribution before resampling
        original_class_distribution = {
            '0': int((y_train == 0).sum()),
            '1': int((y_train == 1).sum())
        }

        # Scaling the features (Standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler and feature names
        joblib.dump(scaler, os.path.join(dir_path, 'scaler.pkl'))
        np.save(os.path.join(dir_path, 'feature_names.npy'), X.columns.to_numpy())

        # Resampling for training data using SMOTETomek (handling imbalance)
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)

        # Resampled class distribution
        resampled_class_distribution = {
            '0': int((y_train_resampled == 0).sum()),
            '1': int((y_train_resampled == 1).sum())
        }

        # Save processed data including resampled data
        np.savez('processed_data_resampled.npz',
                 X_train=X_train_resampled,
                 X_test=X_test_scaled,
                 y_train=y_train_resampled,
                 y_test=y_test)

        # Compute class weights for the classifier training
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_train_resampled
        )

        # Sample for checking (first 5 rows of the resampled data)
        sample_df = pd.DataFrame(X_train_resampled[:5], columns=X.columns)
        sample_df['Class'] = y_train_resampled[:5].values

        # Information about the preprocessing
        return {
            'sample': sample_df.to_dict(orient='records'),
            'info': {
                'message': 'Preprocessing complete with SMOTETomek. The class distribution in training data is balanced.',
                'shapes': {
                    'X_train': X_train_resampled.shape,
                    'X_test': X_test_scaled.shape,
                    'y_train': y_train_resampled.shape,
                    'y_test': y_test.shape
                },
                'class_weights': {
                    '0': float(class_weights[0]),
                    '1': float(class_weights[1])
                },
                'original_class_distribution': original_class_distribution,
                'resampled_class_distribution': resampled_class_distribution
            }
        }
    except Exception as e:
        return {'error': str(e)}
