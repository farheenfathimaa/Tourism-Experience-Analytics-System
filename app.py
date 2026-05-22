import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation_system import get_collaborative_recommendations, get_content_recommendations

# Set page configuration
st.set_page_config(
    page_title="Tourism Experience Analytics System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
def local_css():
    st.markdown("""
    <style>
    /* Font style */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title banner styling */
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .title-banner h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .title-banner p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    section[data-testid="stSidebar"] h2 {
        color: #1e3c72 !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #212529 !important;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1e3c72;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.95rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    
    /* Styled Predictions Output */
    .prediction-container {
        background: #f1f3f9;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #dcdfe6;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .prediction-container h3 {
        color: #212529 !important;
        font-weight: 600;
    }
    .prediction-container p {
        color: #495057 !important;
    }
    .prediction-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3c72;
        margin: 0.5rem 0;
    }
    
    /* Styled Submit Button */
    .stButton>button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(42, 82, 152, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(42, 82, 152, 0.3);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Run CSS styling
local_css()

# Load assets and models
@st.cache_resource
def load_assets():
    try:
        with open("label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open("scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        with open("best_classification_model.pkl", "rb") as f:
            clf_model = pickle.load(f)
        with open("best_regression_model.pkl", "rb") as f:
            reg_model = pickle.load(f)
        with open("item_df.pkl", "rb") as f:
            item_df = pickle.load(f)
        df = pd.read_csv("cleaned_data.csv")
        return encoders, scalers, clf_model, reg_model, item_df, df
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please ensure all preprocessing and model training scripts have run successfully.")
        st.stop()

encoders, scalers, clf_model, reg_model, item_df, df = load_assets()

# Helpers for dynamic selection and prediction mapping
@st.cache_data
def get_demographics_mapping(df_cleaned):
    # Map country to continent, region, and most common CityName
    mapping = {}
    for country, group in df_cleaned.groupby('Country'):
        continent = group['Continent'].iloc[0]
        region = group['Region'].iloc[0]
        cities = group['CityName'].value_counts()
        default_city = cities.index[0] if not cities.empty else 'Unknown'
        mapping[country] = {
            'Continent': continent,
            'Region': region,
            'CityName': default_city
        }
    return mapping

@st.cache_data
def get_continent_countries(df_cleaned):
    # List of countries for each continent
    mapping = {}
    for continent, group in df_cleaned.groupby('Continent'):
        mapping[continent] = sorted(group['Country'].unique().tolist())
    return mapping

@st.cache_data
def get_attractions_by_type(item_metadata):
    # Group attractions by category type
    mapping = {}
    for att_type, group in item_metadata.groupby('AttractionType'):
        mapping[att_type] = group[['AttractionId', 'Attraction']].to_dict('records')
    return mapping

demographics_map = get_demographics_mapping(df)
continent_countries_map = get_continent_countries(df)
attractions_by_type = get_attractions_by_type(item_df)

# Star rating visual generator
def get_star_html(rating):
    rounded = round(rating * 2) / 2
    full_stars = int(rounded)
    half_star = 1 if (rounded - full_stars) == 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    stars = "★" * full_stars + "½" * half_star + "☆" * empty_stars
    return f"<div style='font-size: 28px; color: #FFD700; font-weight: bold;'>{stars} <span style='color: #495057; font-size: 20px;'>({rating:.2f}/5.0)</span></div>"

# Header Banner
st.markdown("""
<div class="title-banner">
    <h1>Tourism Experience Analytics System</h1>
    <p>Classification, Prediction & Personalized Recommendation Engine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("<h2 style='text-align: center; color: #1e3c72; font-weight: 800; margin-bottom: 1.5rem;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["📊 Dashboard & EDA", "🔮 Visit Mode Predictor", "⭐️ Rating Predictor", "🗺️ Attraction Recommender"]
)

# ----------------- PAGE 1: DASHBOARD & EDA -----------------
if page == "📊 Dashboard & EDA":
    st.markdown("<h2 style='color: #1e3c72;'>Executive Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Insights and visual distributions generated from the master dataset of **52,930** user transactions.")
    
    # Key Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Transactions</h3>
            <p>52,930</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <h3>Unique Active Users</h3>
            <p>33,530</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <h3>Available Attractions</h3>
            <p>1,698</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <h3>Overall Average Rating</h3>
            <p>4.16 / 5.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    # Visualizations Grid
    st.markdown("<h3 style='color: #2a5298;'>Dataset Explorations & Chart Gallery</h3>", unsafe_allow_html=True)
    
    # Ensure plots directory has files, else show message
    if not os.path.exists("plots"):
        st.warning("Plots directory not found. Please run eda_visualization.py to generate plots.")
    else:
        # Row 1
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.image("plots/rating_distribution.png", caption="1. User Ratings Distribution", use_column_width=True)
        with r1_c2:
            st.image("plots/visit_mode_distribution.png", caption="2. Visiting Mode Distribution", use_column_width=True)
            
        # Row 2
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.image("plots/top_attractions.png", caption="3. Top 10 Most Visited Attractions", use_column_width=True)
        with r2_c2:
            st.image("plots/user_continent_distribution.png", caption="4. User Distribution by Continent", use_column_width=True)
            
        # Row 3
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            st.image("plots/monthly_trends.png", caption="5. Monthly Visit Trends (Seasonality)", use_column_width=True)
        with r3_c2:
            st.image("plots/avg_rating_by_type.png", caption="6. Average Rating by Attraction Type", use_column_width=True)
            
        # Row 4
        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            st.image("plots/correlation_heatmap.png", caption="7. Feature Correlation Heatmap", use_column_width=True)
        with r4_c2:
            st.image("plots/rating_by_visit_mode_boxplot.png", caption="8. Rating Distribution by Visit Mode", use_column_width=True)

# ----------------- PAGE 2: VISIT MODE PREDICTOR -----------------
elif page == "🔮 Visit Mode Predictor":
    st.markdown("<h2 style='color: #1e3c72;'>Predict Visiting Mode</h2>", unsafe_allow_html=True)
    st.markdown("Classifies user traveling type (Business, Couples, Family, Friends, Solo) based on demographic, timing, attraction type, and expected rating.")
    
    st.write("---")
    
    # Input Form Layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographics
        continent = st.selectbox("Select User Continent", sorted(list(continent_countries_map.keys())))
        
        # Filter countries based on continent
        available_countries = continent_countries_map[continent]
        country = st.selectbox("Select User Country", available_countries)
        
        # Sliders for year/month
        visit_month = st.slider("Select Month of Visit", min_value=1, max_value=12, value=6, step=1)
        visit_year = st.slider("Select Year of Visit", min_value=2015, max_value=2025, value=2022, step=1)
        
    with col2:
        # Attraction attributes
        all_att_types = sorted(list(attractions_by_type.keys()))
        att_type = st.selectbox("Select Attraction Type", all_att_types)
        
        # Filter attractions based on type to fetch a valid AttractionId
        att_list = attractions_by_type[att_type]
        attraction_names = [x['Attraction'] for x in att_list]
        selected_att_name = st.selectbox("Select Attraction", attraction_names)
        
        # Get AttractionId
        selected_att_id = [x['AttractionId'] for x in att_list if x['Attraction'] == selected_att_name][0]
        
        # Rating slider
        rating = st.slider("Rating Given/Expected", min_value=1, max_value=5, value=4, step=1)
        
    # Trigger Prediction
    if st.button("Predict Visit Mode"):
        # Retrieve region and city from country mapping
        demographics = demographics_map[country]
        region = demographics['Region']
        city_name = demographics['CityName']
        
        # Encode values
        try:
            continent_enc = encoders['Continent'].transform([continent])[0]
            region_enc = encoders['Region'].transform([region])[0]
            country_enc = encoders['Country'].transform([country])[0]
            city_enc = encoders['CityName'].transform([city_name])[0]
            att_type_enc = encoders['AttractionType'].transform([att_type])[0]
            
            # Scale numerical values
            year_scaler = scalers['VisitYear']
            month_scaler = scalers['VisitMonth']
            rating_scaler = scalers['Rating']
            
            year_scaled = year_scaler.transform([[visit_year]])[0][0]
            month_scaled = month_scaler.transform([[visit_month]])[0][0]
            rating_scaled = rating_scaler.transform([[rating]])[0][0]
            
            # Construct Feature Input Array
            # ['Continent_encoded', 'Region_encoded', 'Country_encoded', 'CityName_encoded', 'VisitYear_scaled', 'VisitMonth_scaled', 'AttractionId', 'AttractionType_encoded', 'Rating_scaled']
            features_input = np.array([[
                continent_enc, region_enc, country_enc, city_enc,
                year_scaled, month_scaled, selected_att_id, att_type_enc, rating_scaled
            ]])
            
            # Predict
            pred_idx = clf_model.predict(features_input)[0]
            pred_mode = encoders['VisitMode'].inverse_transform([pred_idx])[0]
            
            # Predict probabilities for chart
            probs = clf_model.predict_proba(features_input)[0]
            class_indices = clf_model.classes_
            class_names = encoders['VisitMode'].inverse_transform(class_indices)
            
            st.markdown("### Prediction Results")
            st.markdown(f"""
            <div class="prediction-container">
                <h3>Predicted Travel Segment / Visit Mode</h3>
                <div class="prediction-value">{pred_mode}</div>
                <p>Calculated dynamically based on demographic history and attraction profiles</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot Confidence bar chart
            st.write("---")
            st.markdown("#### Prediction Confidence Distribution")
            prob_df = pd.DataFrame({
                'Visit Mode': class_names,
                'Probability (%)': [round(p * 100, 2) for p in probs]
            }).sort_values(by='Probability (%)', ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=prob_df, x='Probability (%)', y='Visit Mode', palette="viridis", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence Percentage (%)")
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error making classification prediction: {e}")

# ----------------- PAGE 3: RATING PREDICTOR -----------------
elif page == "⭐️ Rating Predictor":
    st.markdown("<h2 style='color: #1e3c72;'>Predict Attraction Rating</h2>", unsafe_allow_html=True)
    st.markdown("Predicts the numerical rating (1.0 to 5.0) that a user would give to an attraction.")
    
    st.write("---")
    
    # Input Form Layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographics
        continent = st.selectbox("Select User Continent", sorted(list(continent_countries_map.keys())))
        available_countries = continent_countries_map[continent]
        country = st.selectbox("Select User Country", available_countries)
        
        # Sliders for year/month
        visit_month = st.slider("Select Month of Visit", min_value=1, max_value=12, value=6, step=1)
        visit_year = st.slider("Select Year of Visit", min_value=2015, max_value=2025, value=2022, step=1)
        
    with col2:
        # Mode & Type attributes
        all_visit_modes = [x for x in list(encoders['VisitMode'].classes_) if x != "Unknown"]
        visit_mode = st.selectbox("Select Traveling Mode", all_visit_modes)
        
        all_att_types = sorted(list(attractions_by_type.keys()))
        att_type = st.selectbox("Select Attraction Type", all_att_types)
        
    # Trigger Prediction
    if st.button("Predict Expected Rating"):
        # Retrieve region and city from country mapping
        demographics = demographics_map[country]
        region = demographics['Region']
        city_name = demographics['CityName']
        
        # Encode values
        try:
            continent_enc = encoders['Continent'].transform([continent])[0]
            region_enc = encoders['Region'].transform([region])[0]
            country_enc = encoders['Country'].transform([country])[0]
            city_enc = encoders['CityName'].transform([city_name])[0]
            att_type_enc = encoders['AttractionType'].transform([att_type])[0]
            visit_mode_enc = encoders['VisitMode'].transform([visit_mode])[0]
            
            # Scale numerical values
            year_scaler = scalers['VisitYear']
            month_scaler = scalers['VisitMonth']
            
            year_scaled = year_scaler.transform([[visit_year]])[0][0]
            month_scaled = month_scaler.transform([[visit_month]])[0][0]
            
            # Construct Feature Input Array
            # ['Continent_encoded', 'Region_encoded', 'Country_encoded', 'CityName_encoded', 'VisitMonth_scaled', 'VisitYear_scaled', 'AttractionType_encoded', 'VisitMode_encoded']
            features_input = np.array([[
                continent_enc, region_enc, country_enc, city_enc,
                month_scaled, year_scaled, att_type_enc, visit_mode_enc
            ]])
            
            # Predict
            pred_rating = float(reg_model.predict(features_input)[0])
            pred_rating = np.clip(pred_rating, 1.0, 5.0) # bound within realistic ratings
            
            st.markdown("### Prediction Results")
            st.markdown(f"""
            <div class="prediction-container">
                <h3>Predicted Score / Rating (Scale 1-5)</h3>
                <div style="margin: 1rem 0;">{get_star_html(pred_rating)}</div>
                <p>Expected satisfaction score predicted based on user profiles and location attractions</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making regression prediction: {e}")

# ----------------- PAGE 4: ATTRACTION RECOMMENDER -----------------
elif page == "🗺️ Attraction Recommender":
    st.markdown("<h2 style='color: #1e3c72;'>Tourism Recommendation Engine</h2>", unsafe_allow_html=True)
    st.markdown("Utilize **User-User Collaborative Filtering** to recommend new attractions based on user profiles or use **Content-Based Filtering** to find similar attractions.")
    
    st.write("---")
    
    # Select Recommender Approach
    tab1, tab2 = st.tabs(["👥 User-Based Recommendations (Collaborative)", "🎭 Attraction similarity (Content-Based)"])
    
    with tab1:
        st.markdown("### Personalized User Recommendations")
        st.markdown("Enter a UserId to find top tourist attractions that similar users highly rated, but this user has not visited yet.")
        
        # Display sample User IDs to user for convenience
        st.info("💡 **Sample User IDs to try:** 70456, 7567, 14, 16, 117, 331, 452")
        
        user_input_id = st.text_input("Enter User ID", value="70456")
        
        if st.button("Generate Collaborative Recommendations"):
            if not user_input_id.strip():
                st.warning("Please enter a valid User ID.")
            else:
                try:
                    uid = int(user_input_id.strip())
                    recs_df = get_collaborative_recommendations(uid, top_n=5)
                    
                    if recs_df.empty:
                        st.info("No recommendations found or User ID has no historical context.")
                    else:
                        st.markdown("#### Recommended Attractions for You:")
                        st.table(recs_df)
                except ValueError:
                    st.error("User ID must be a numerical integer.")
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    
    with tab2:
        st.markdown("### Similarity-Based Recommendations")
        st.markdown("Select an attraction from the dropdown to discover the top 5 most similar attractions based on Attraction Category Type and Location City.")
        
        # Dropdown listing all attractions alphabetically
        sorted_attractions = item_df[['AttractionId', 'Attraction']].sort_values(by='Attraction')
        att_options = sorted_attractions['Attraction'].tolist()
        selected_att_opt = st.selectbox("Select Target Attraction", att_options)
        
        # Get corresponding AttractionId
        target_att_id = sorted_attractions[sorted_attractions['Attraction'] == selected_att_opt].iloc[0]['AttractionId']
        
        if st.button("Find Similar Attractions"):
            try:
                recs_df = get_content_recommendations(target_att_id, top_n=5)
                
                if recs_df.empty:
                    st.info("No similar attractions found.")
                else:
                    st.markdown(f"#### Top 5 Attractions similar to **{selected_att_opt}**:")
                    st.table(recs_df[['Rank', 'Attraction', 'AttractionType', 'AttractionCity', 'SimilarityScore', 'Address']])
            except Exception as e:
                st.error(f"Error generating content-based recommendations: {e}")
