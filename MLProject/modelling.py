"""
Modelling for MLflow Project CI - Kendrick Filbert
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')


def main():
    mlflow.set_experiment("Breast_Cancer_CI")

    train_df = pd.read_csv("breast_cancer_train_preprocessing.csv")
    test_df = pd.read_csv("breast_cancer_test_preprocessing.csv")
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    print(f"[INFO] Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    print("[INFO] Starting GridSearchCV...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    with mlflow.start_run(run_name="RF_CI_Pipeline"):
        for k, v in grid_search.best_params_.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # Feature importance artifact
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        mlflow.log_artifact("feature_importance.png")

        mlflow.set_tag("author", "Kendrick Filbert")
        mlflow.set_tag("pipeline", "CI")

        print("\n" + "=" * 50)
        print("CI PIPELINE - TRAINING COMPLETE")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:20s}: {v:.4f}")
        print(f"  Run ID: {mlflow.active_run().info.run_id}")

    for f in ["confusion_matrix.png", "feature_importance.png"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
