import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json

# Set MLflow experiment
mlflow.set_experiment("classification_experiment")

def train_model(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, 
                n_estimators=100, max_depth=10, random_state=42):
    """
    Train a classification model with specified parameters and log results to MLflow
    """
    # Validate parameters to avoid ValueError
    if n_informative + n_redundant >= n_features:
        print(f"Warning: Adjusting n_redundant to ensure valid parameters")
        n_redundant = max(0, n_features - n_informative - 1)  # Ensure valid parameters
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    joblib.dump(scaler, "scaler.pkl")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log parameters and metrics with MLflow
    with mlflow.start_run() as run:
        # Log data parameters
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("n_informative", n_informative)
        mlflow.log_param("n_redundant", n_redundant)
        
        # Log model parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save feature importance
        feature_importance = model.feature_importances_
        feature_importance_dict = {f"feature_{i}": float(importance) 
                                  for i, importance in enumerate(feature_importance)}
        with open("feature_importance.json", "w") as f:
            json.dump(feature_importance_dict, f)
        mlflow.log_artifact("feature_importance.json")
        
        # Save test data for later use
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)
        
        # Save run_id to a file
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

            
    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "run_id": run_id  # Use the saved run_id variable
    }


if __name__ == "__main__":
    # Example parameter tuning with more conservative parameter ranges
    n_samples_list = [1000, 2000]
    n_features_list = [20, 30]
    n_informative_list = [8, 12]
    n_redundant_list = [4, 8]
    n_estimators_list = [100, 200]
    max_depth_list = [10, 15]
    
    best_f1 = 0
    best_run_id = None
    
    # Run experiments with different parameters
    for n_samples in n_samples_list:
        for n_features in n_features_list:
            for n_informative in n_informative_list:
                for n_redundant in n_redundant_list:
                    # Skip invalid combinations
                    if n_informative + n_redundant >= n_features:
                        print(f"Skipping invalid combination: features={n_features}, "
                              f"informative={n_informative}, redundant={n_redundant}")
                        continue
                        
                    for n_estimators in n_estimators_list:
                        for max_depth in max_depth_list:
                            print(f"Training with: samples={n_samples}, features={n_features}, "
                                  f"informative={n_informative}, redundant={n_redundant}, "
                                  f"estimators={n_estimators}, max_depth={max_depth}")
                            
                            result = train_model(
                                n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                n_redundant=n_redundant,
                                n_estimators=n_estimators,
                                max_depth=max_depth
                            )
                            
                            # Track best model based on F1 score
                            if result["f1"] > best_f1:
                                best_f1 = result["f1"]
                                best_run_id = result["run_id"]
                                
    print(f"Best model run_id: {best_run_id} with F1 score: {best_f1}")
    
    # Save best run ID to a file for the web app to use
    with open("best_run_id.txt", "w") as f:
        f.write(best_run_id)