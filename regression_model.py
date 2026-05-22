import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def main():
    print("--- STEP 1: Loading Cleaned Data ---")
    if not os.path.exists("cleaned_data.csv"):
        raise FileNotFoundError("cleaned_data.csv not found. Please run data_preprocessing.py first.")
        
    df = pd.read_csv("cleaned_data.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")

    print("\n--- STEP 2: Preparing Features and Target ---")
    # Features as specified in instructions:
    # All user demographic features + VisitMonth + VisitYear + AttractionType + encoded VisitMode
    features = [
        'Continent_encoded', 
        'Region_encoded', 
        'Country_encoded', 
        'CityName_encoded', 
        'VisitMonth_scaled', 
        'VisitYear_scaled', 
        'AttractionType_encoded', 
        'VisitMode_encoded'
    ]
    target = 'Rating'

    X = df[features]
    y = df[target]

    print("Features used:", features)
    print("Target:", target)

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    print("\n--- STEP 3: Training Models ---")
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            "Model": name,
            "R2": r2,
            "MSE": mse,
            "MAE": mae
        })
        trained_models[name] = model

    # Create Comparison DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    # Determine best model based on lowest MSE
    best_row = results_df.loc[results_df['MSE'].idxmin()]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    
    print(f"\nBest Model: {best_model_name} with MSE of {best_row['MSE']:.5f}")

    # Save the best model
    model_filename = "best_regression_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved best model to {model_filename}")

if __name__ == "__main__":
    main()
