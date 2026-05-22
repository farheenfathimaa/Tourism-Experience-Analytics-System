import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def precompute_content_based():
    print("\n--- STEP 1: Precomputing Content-Based Similarity (1698 attractions) ---")
    dataset_dir = "Tourism Dataset"
    
    # Load items and decode type and city
    item = pd.read_excel(os.path.join(dataset_dir, "Item.xlsx"))
    updated_item = pd.read_excel(os.path.join(dataset_dir, "Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx"))
    
    # Combine and deduplicate
    att = pd.concat([item, updated_item], ignore_index=True)
    att = att.drop_duplicates(subset=['AttractionId']).copy()
    
    # Decode AttractionType and AttractionCity
    type_df = pd.read_excel(os.path.join(dataset_dir, "Type.xlsx"))
    city_df = pd.read_excel(os.path.join(dataset_dir, "City.xlsx"))
    
    att = pd.merge(att, type_df, on="AttractionTypeId", how="left")
    att = pd.merge(att, city_df[['CityId', 'CityName']], left_on="AttractionCityId", right_on="CityId", how="left")
    att = att.rename(columns={'CityName': 'AttractionCity'})
    
    # Handle NaNs and placeholders
    att['AttractionType'] = att['AttractionType'].fillna("Unknown").replace("-", "Unknown")
    att['AttractionCity'] = att['AttractionCity'].fillna("Unknown").replace("-", "Unknown")
    att['Attraction'] = att['Attraction'].fillna("Unknown")
    att['AttractionAddress'] = att['AttractionAddress'].fillna("Unknown")
    
    # One-hot encode features for similarity
    features = pd.get_dummies(att[['AttractionType', 'AttractionCity']])
    
    # Compute Cosine Similarity
    content_sim = cosine_similarity(features.values)
    
    print(f"Content-Based matrix shape: {content_sim.shape}")
    
    # Save files
    with open("content_similarity.pkl", "wb") as f:
        pickle.dump(content_sim, f)
    with open("item_df.pkl", "wb") as f:
        pickle.dump(att, f)
    print("Saved content_similarity.pkl and item_df.pkl")
    return att, content_sim

def evaluate_collaborative_filtering(df):
    print("\n--- STEP 2: Evaluating Collaborative Filtering ---")
    
    # Get transaction data and average ratings for duplicate (UserId, AttractionId) pairs
    tx = df.groupby(['UserId', 'AttractionId'])['Rating'].mean().reset_index()
    
    # Train-test split (80/20)
    train_tx, test_tx = train_test_split(tx, test_size=0.2, random_state=42)
    print(f"Train transactions: {len(train_tx)}, Test transactions: {len(test_tx)}")
    
    # Create user-item matrix on train set
    user_item_train = train_tx.pivot(index='UserId', columns='AttractionId', values='Rating')
    
    # Calculate user means (for centering ratings and filling missing values)
    user_means = user_item_train.mean(axis=1)
    global_mean = train_tx['Rating'].mean()
    
    # Center user ratings (subtract mean) and fill NaN with 0
    user_item_centered = user_item_train.sub(user_means, axis=0).fillna(0)
    
    print("Evaluating collaborative filtering on test set (using sample of 2000 test ratings for speed)...")
    # Sample test set for evaluation to ensure fast execution
    test_sample = test_tx.sample(n=min(2000, len(test_tx)), random_state=42)
    
    actuals = []
    preds = []
    
    # Convert index for fast lookup
    user_indices = {uid: i for i, uid in enumerate(user_item_centered.index)}
    centered_matrix = user_item_centered.values # N_users x N_items
    attraction_ids = list(user_item_train.columns)
    attraction_indices = {aid: i for i, aid in enumerate(attraction_ids)}
    
    # Pre-calculate user norms for cosine similarity
    norms = np.linalg.norm(centered_matrix, axis=1)
    norms[norms == 0] = 1e-9 # avoid division by zero
    
    for idx, row in test_sample.iterrows():
        uid = int(row['UserId'])
        aid = int(row['AttractionId'])
        actual_rating = row['Rating']
        
        # Check if user and item are in training set
        if uid not in user_indices or aid not in attraction_indices:
            pred_rating = global_mean
        else:
            u_idx = user_indices[uid]
            a_idx = attraction_indices[aid]
            
            # Find users who have rated this attraction in train set
            # Those are users where user_item_train[aid] is not NaN
            rating_column = user_item_train[aid]
            rated_user_ids = rating_column.dropna().index
            
            if len(rated_user_ids) == 0:
                pred_rating = user_means.get(uid, global_mean)
            else:
                rated_user_indices = [user_indices[ruid] for ruid in rated_user_ids]
                
                # Compute cosine similarity between query user and these users
                # Cosine similarity = dot(u, v) / (norm(u) * norm(v))
                u_vec = centered_matrix[u_idx]
                v_vecs = centered_matrix[rated_user_indices]
                
                dots = np.dot(v_vecs, u_vec)
                sims = dots / (norms[u_idx] * norms[rated_user_indices])
                
                # Filter negative similarities and keep top K
                pos_indices = np.where(sims > 0)[0]
                if len(pos_indices) == 0:
                    pred_rating = user_means.get(uid, global_mean)
                else:
                    sims = sims[pos_indices]
                    ratings = rating_column.loc[rated_user_ids[pos_indices]].values
                    
                    # Top 10 similar
                    top_k = min(10, len(sims))
                    top_k_idx = np.argsort(sims)[::-1][:top_k]
                    
                    top_sims = sims[top_k_idx]
                    top_ratings = ratings[top_k_idx]
                    
                    sim_sum = np.sum(top_sims)
                    if sim_sum == 0:
                        pred_rating = user_means.get(uid, global_mean)
                    else:
                        pred_rating = np.sum(top_sims * top_ratings) / sim_sum
                        
        actuals.append(actual_rating)
        preds.append(pred_rating)
        
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"Collaborative Filtering Test RMSE: {rmse:.4f}")
    return rmse

def save_collaborative_filtering_data(df):
    print("\n--- STEP 3: Saving Collaborative Filtering Data structures ---")
    
    # Build user-item rating matrix on the ENTIRE dataset for production recommendations
    # Group by and average ratings for duplicate (UserId, AttractionId) pairs
    tx = df.groupby(['UserId', 'AttractionId'])['Rating'].mean().reset_index()
    user_item_matrix = tx.pivot(index='UserId', columns='AttractionId', values='Rating')
    
    # Save the user-item matrix
    with open("user_item_matrix.pkl", "wb") as f:
        pickle.dump(user_item_matrix, f)
        
    print("Saved user_item_matrix.pkl")

# Recommendation Functions Exposed for the Streamlit App
def get_collaborative_recommendations(user_id, top_n=5):
    # Load user-item matrix and attraction metadata
    with open("user_item_matrix.pkl", "rb") as f:
        user_item = pickle.load(f)
    with open("item_df.pkl", "rb") as f:
        item_df = pickle.load(f)
        
    # Check if user exists in training set
    if user_id not in user_item.index:
        # Cold start: return top-N overall highest rated attractions
        print(f"User {user_id} not found. Returning popular attractions.")
        popular = item_df.head(top_n).copy()
        popular['RecommendScore'] = 5.0
        return popular[['AttractionId', 'Attraction', 'AttractionType', 'AttractionCity', 'RecommendScore']]
        
    # Center the matrix
    user_means = user_item.mean(axis=1)
    user_item_centered = user_item.sub(user_means, axis=0).fillna(0)
    
    user_indices = {uid: i for i, uid in enumerate(user_item_centered.index)}
    centered_matrix = user_item_centered.values
    attraction_ids = list(user_item.columns)
    attraction_indices = {aid: i for i, aid in enumerate(attraction_ids)}
    norms = np.linalg.norm(centered_matrix, axis=1)
    norms[norms == 0] = 1e-9
    
    u_idx = user_indices[user_id]
    u_vec = centered_matrix[u_idx]
    
    # Compute similarity with all other users
    dots = np.dot(centered_matrix, u_vec)
    sims = dots / (norms[u_idx] * norms)
    
    # Get attractions user has NOT visited
    user_ratings = user_item.loc[user_id]
    unvisited_ids = user_ratings[user_ratings.isna()].index
    
    predictions = []
    
    for aid in unvisited_ids:
        # Find users who rated this attraction
        rating_col = user_item[aid]
        rated_users = rating_col.dropna().index
        
        if len(rated_users) == 0:
            continue
            
        rated_u_indices = [user_indices[ruid] for ruid in rated_users]
        user_sims = sims[rated_u_indices]
        
        # Keep positive similarities
        pos_sim_indices = np.where(user_sims > 0)[0]
        if len(pos_sim_indices) == 0:
            continue
            
        pos_sims = user_sims[pos_sim_indices]
        pos_ratings = rating_col.loc[rated_users[pos_sim_indices]].values
        
        # Weighted average rating of top 10 similar users
        top_k = min(10, len(pos_sims))
        top_k_idx = np.argsort(pos_sims)[::-1][:top_k]
        
        top_sims = pos_sims[top_k_idx]
        top_ratings = pos_ratings[top_k_idx]
        
        sim_sum = np.sum(top_sims)
        if sim_sum > 0:
            pred_score = np.sum(top_sims * top_ratings) / sim_sum
            predictions.append((aid, pred_score))
            
    # Sort predictions
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Match with attraction names
    recs = []
    for rank, (aid, score) in enumerate(predictions):
        row = item_df[item_df['AttractionId'] == aid]
        if not row.empty:
            recs.append({
                "Rank": rank + 1,
                "AttractionId": aid,
                "Attraction": row.iloc[0]['Attraction'],
                "AttractionType": row.iloc[0]['AttractionType'],
                "AttractionCity": row.iloc[0]['AttractionCity'],
                "RecommendScore": round(score, 2)
            })
            
    return pd.DataFrame(recs)

def get_content_recommendations(attraction_id, top_n=5):
    # Load content-based matrices
    with open("content_similarity.pkl", "rb") as f:
        content_sim = pickle.load(f)
    with open("item_df.pkl", "rb") as f:
        item_df = pickle.load(f)
        
    # Find index of attraction
    matching_rows = item_df[item_df['AttractionId'] == attraction_id]
    if matching_rows.empty:
        print(f"Attraction ID {attraction_id} not found.")
        return pd.DataFrame()
        
    idx = matching_rows.index[0]
    
    # Get similarity scores for this attraction
    sim_scores = list(enumerate(content_sim[idx]))
    
    # Sort based on similarity, excluding the attraction itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if item_df.iloc[x[0]]['AttractionId'] != attraction_id][:top_n]
    
    recs = []
    for rank, (item_idx, score) in enumerate(sim_scores):
        row = item_df.iloc[item_idx]
        recs.append({
            "Rank": rank + 1,
            "AttractionId": row['AttractionId'],
            "Attraction": row['Attraction'],
            "AttractionType": row['AttractionType'],
            "AttractionCity": row['AttractionCity'],
            "SimilarityScore": round(score, 3),
            "Address": row['AttractionAddress']
        })
        
    return pd.DataFrame(recs)

def main():
    # 1. Precompute content based recommendations
    precompute_content_based()
    
    # Load cleaned data for collaborative filtering
    if not os.path.exists("cleaned_data.csv"):
        raise FileNotFoundError("cleaned_data.csv not found. Please run data_preprocessing.py first.")
    df = pd.read_csv("cleaned_data.csv")
    
    # 2. Evaluate collaborative filtering
    evaluate_collaborative_filtering(df)
    
    # 3. Save collaborative filtering matrix
    save_collaborative_filtering_data(df)
    
    print("\nRecommendation system initialized and matrices saved successfully.")

if __name__ == "__main__":
    main()
