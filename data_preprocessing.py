import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def main():
    dataset_dir = "Tourism Dataset"
    print("--- STEP 1: Loading all 10 Excel files ---")
    
    # Check paths
    files_to_check = [
        "Transaction.xlsx", "User.xlsx", "Item.xlsx",
        "City.xlsx", "Country.xlsx", "Region.xlsx", "Continent.xlsx",
        "Type.xlsx", "Mode.xlsx",
        os.path.join("Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx")
    ]
    
    for f in files_to_check:
        path = os.path.join(dataset_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required Excel file not found: {path}")

    # Load dataframes
    tx = pd.read_excel(os.path.join(dataset_dir, "Transaction.xlsx"))
    user = pd.read_excel(os.path.join(dataset_dir, "User.xlsx"))
    item = pd.read_excel(os.path.join(dataset_dir, "Item.xlsx"))
    updated_item = pd.read_excel(os.path.join(dataset_dir, "Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx"))
    
    city = pd.read_excel(os.path.join(dataset_dir, "City.xlsx"))
    country = pd.read_excel(os.path.join(dataset_dir, "Country.xlsx"))
    region = pd.read_excel(os.path.join(dataset_dir, "Region.xlsx"))
    continent = pd.read_excel(os.path.join(dataset_dir, "Continent.xlsx"))
    att_type = pd.read_excel(os.path.join(dataset_dir, "Type.xlsx"))
    mode = pd.read_excel(os.path.join(dataset_dir, "Mode.xlsx"))

    print(f"Transaction shape: {tx.shape}")
    print(f"User shape: {user.shape}")
    print(f"Item shape: {item.shape}")
    print(f"Updated Item shape: {updated_item.shape}")

    # Combine Item and Updated Item to ensure full attraction database
    combined_item = pd.concat([item, updated_item], ignore_index=True)
    combined_item = combined_item.drop_duplicates(subset=['AttractionId'])
    print(f"Combined and deduplicated Items shape: {combined_item.shape}")

    # Record null count before preprocessing
    total_nulls_before = tx.isnull().sum().sum() + user.isnull().sum().sum() + combined_item.isnull().sum().sum()

    print("\n--- STEP 2: Merging dataframes and decoding IDs ---")
    
    # Handle missing CityId in User by setting it to 0 (which maps to "-" in City lookup, subsequently handled as "Unknown")
    user['CityId'] = user['CityId'].fillna(0).astype(int)

    # 1. Merge Transaction with User
    merged = pd.merge(tx, user, on="UserId", how="left")
    
    # 2. Merge with combined Item
    merged = pd.merge(merged, combined_item, on="AttractionId", how="left")

    # Rename VisitMode in Transaction to match VisitModeId in Mode lookup
    merged = merged.rename(columns={'VisitMode': 'VisitModeId'})

    # Prune lookup tables to only include required columns to avoid conflicts (e.g. RegionId, ContinentId)
    mode_lookup = mode[['VisitModeId', 'VisitMode']].copy()
    continent_lookup = continent[['ContinentId', 'Continent']].copy()
    region_lookup = region[['RegionId', 'Region']].copy()
    country_lookup = country[['CountryId', 'Country']].copy()
    city_user_lookup = city[['CityId', 'CityName']].copy()
    type_lookup = att_type[['AttractionTypeId', 'AttractionType']].copy()
    city_attraction_lookup = city[['CityId', 'CityName']].rename(columns={'CityId': 'AttractionCityId', 'CityName': 'AttractionCity'}).copy()

    # 3. Decode VisitModeId
    merged = pd.merge(merged, mode_lookup, on="VisitModeId", how="left")
    # 4. Decode ContinentId
    merged = pd.merge(merged, continent_lookup, on="ContinentId", how="left")
    # 5. Decode RegionId
    merged = pd.merge(merged, region_lookup, on="RegionId", how="left")
    # 6. Decode CountryId
    merged = pd.merge(merged, country_lookup, on="CountryId", how="left")
    # 7. Decode CityId (User's City)
    merged = pd.merge(merged, city_user_lookup, on="CityId", how="left")
    # 8. Decode AttractionTypeId
    merged = pd.merge(merged, type_lookup, on="AttractionTypeId", how="left")
    # 9. Decode AttractionCityId
    merged = pd.merge(merged, city_attraction_lookup, on="AttractionCityId", how="left")

    # Drop intermediate lookup ID columns that have been decoded
    cols_to_drop = [
        'VisitModeId', 'ContinentId', 'RegionId', 'CountryId', 'CityId',
        'AttractionCityId', 'AttractionTypeId'
    ]
    merged = merged.drop(columns=cols_to_drop, errors='ignore')

    print("\n--- STEP 3: Cleaning missing/null values and placeholder '-' ---")
    
    # Categorical columns to clean and process
    categorical_cols = [
        'VisitMode', 'Continent', 'Region', 'Country', 'CityName',
        'AttractionType', 'Attraction', 'AttractionCity', 'AttractionAddress'
    ]
    
    # Replace NaN and "-" placeholders with "Unknown"
    for col in categorical_cols:
        merged[col] = merged[col].fillna("Unknown")
        merged[col] = merged[col].replace("-", "Unknown")
        merged[col] = merged[col].astype(str).str.strip()

    # Handle numerical nulls if any
    numerical_cols = ['Rating', 'VisitYear', 'VisitMonth']
    for col in numerical_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        # Fill missing values with median if there are any
        if merged[col].isnull().any():
            merged[col] = merged[col].fillna(merged[col].median())

    print("\n--- STEP 4: Feature Engineering (User Aggregates) ---")
    
    # Create user-level aggregates
    # 1. Average rating per user
    user_avg_rating = merged.groupby('UserId')['Rating'].transform('mean')
    merged['UserAvgRating'] = user_avg_rating
    
    # 2. Visit count per user
    user_visit_count = merged.groupby('UserId')['Rating'].transform('count')
    merged['UserVisitCount'] = user_visit_count

    print("\n--- STEP 5: Categorical Encoding ---")
    
    # Label encode all categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on all possible values (and add "Unknown" explicitly to classes to handle unseen inputs)
        unique_vals = list(merged[col].unique())
        if "Unknown" not in unique_vals:
            unique_vals.append("Unknown")
        le.fit(unique_vals)
        merged[f"{col}_encoded"] = le.transform(merged[col])
        label_encoders[col] = le
        print(f"  Encoded '{col}' -> classes: {len(le.classes_)}")

    # Save the label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Saved label_encoders.pkl")

    print("\n--- STEP 6: Numerical Feature Scaling ---")
    
    # Numerical features to scale
    numerical_features = ['Rating', 'VisitYear', 'VisitMonth', 'UserAvgRating', 'UserVisitCount']
    scalers = {}
    for col in numerical_features:
        scaler = MinMaxScaler()
        # Scale
        merged[f"{col}_scaled"] = scaler.fit_transform(merged[[col]])
        scalers[col] = scaler
        print(f"  Scaled '{col}' -> min: {scaler.data_min_[0]}, max: {scaler.data_max_[0]}")

    # Save the scalers
    with open('scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print("Saved scalers.pkl")

    # Record null count after preprocessing
    total_nulls_after = merged.isnull().sum().sum()

    print("\n--- STEP 7: Saving Cleaned Data ---")
    merged.to_csv("cleaned_data.csv", index=False)
    print("Saved cleaned_data.csv")

    # Summary Output
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Master Dataset Shape: {merged.shape}")
    print(f"Total Nulls Before Merging: {total_nulls_before}")
    print(f"Total Nulls After Preprocessing: {total_nulls_after}")
    print("\nAll Column Names in Cleaned Dataset:")
    print(merged.columns.tolist())
    print("="*50)

if __name__ == "__main__":
    main()
