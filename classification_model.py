import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():
    print("--- STEP 1: Loading Cleaned Data ---")
    if not os.path.exists("cleaned_data.csv"):
        raise FileNotFoundError("cleaned_data.csv not found. Please run data_preprocessing.py first.")
        
    df = pd.read_csv("cleaned_data.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")

    print("\n--- STEP 2: Preparing Features and Target ---")
    # Features specified:
    # User demographics + VisitYear + VisitMonth + AttractionId + AttractionType + Rating
    features = [
        'Continent_encoded', 
        'Region_encoded', 
        'Country_encoded', 
        'CityName_encoded', 
        'VisitYear_scaled', 
        'VisitMonth_scaled', 
        'AttractionId', 
        'AttractionType_encoded', 
        'Rating_scaled'
    ]
    target = 'VisitMode_encoded'

    X = df[features]
    y = df[target]

    print("Features used:", features)
    print("Target:", target)

    # Train/Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("\n--- STEP 3: Training Models ---")
    
    # Initialize models
    # VisitMode has 5 classes (representing 0 to 4)
    models = {
        "Random Forest Classifier": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "LightGBM Classifier": LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score (macro)": f1
        })
        trained_models[name] = model

    # Create Comparison DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # Determine best model based on F1-score (macro)
    best_row = results_df.loc[results_df['F1-score (macro)'].idxmax()]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with F1-score (macro) of {best_row['F1-score (macro)']:.5f}")

    # Load label encoder to decode the classes for report
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    visit_mode_encoder = encoders['VisitMode']
    target_names = list(visit_mode_encoder.classes_)
    # Filter target names to match actual present labels in model (since "Unknown" might not be in training targets)
    unique_train_targets = sorted(y.unique())
    class_names = [target_names[i] for i in unique_train_targets]

    # Evaluate best model further
    y_pred_best = best_model.predict(X_test)
    
    print("\n" + "="*70)
    print(f"CLASSIFICATION REPORT FOR BEST MODEL ({best_model_name})")
    print("="*70)
    print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))
    print("="*70)

    print("\n" + "="*70)
    print(f"CONFUSION MATRIX FOR BEST MODEL ({best_model_name})")
    print("="*70)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())
    print("="*70)

    # Save the best model
    model_filename = "best_classification_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved best model to {model_filename}")

if __name__ == "__main__":
    main()
