import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Athlete360 Dashboard",
    page_icon="🏀",
    layout="wide"
)

@st.cache_resource
def load_models():
    lgbm_model = joblib.load('lgbm_performance_model.joblib')
    rf_injury_model = joblib.load('rf_injury_model.joblib')
    kmeans_model = joblib.load('kmeans_cluster_model.joblib')
    cluster_scaler = joblib.load('cluster_scaler.joblib')
    return lgbm_model, rf_injury_model, kmeans_model, cluster_scaler

lgbm_model, rf_injury_model, kmeans_model, cluster_scaler = load_models()


def feature_engineer(df):

    df['rolling_points_mean'] = df.groupby('player_name')['points'].transform(lambda x: x.shift(1).rolling(5).mean())
    # ... add all other feature engineering steps ...
    
    # Define the feature lists your models expect
    perf_features = [
        'minutes_played', 'days_since_last_game', 'back_to_back_games', 
        'rolling_points_mean', 'rolling_points_std', # ...and so on
        'fatigue_index', 'opponent_def_rating' 
    ] # List from Q1
    
    injury_features = [
        'minutes_played', 'points', 'assists', 'total_rebounds',
        'turnovers', 'fatigue_index', # ...and so on
    ] # List from Q2

    cluster_features = [
        'rolling_points_mean', 'rolling_points_std',
        'rolling_assists_mean', 'rolling_assists_std', #...and so on
    ] # List from Q3
    
    # For this demo, we'll assume the features are already in the CSV
    # In a real app, you would calculate them here.
    
    # Ensure all required columns exist, fillna if necessary
    # ...
    
    return df, perf_features, injury_features, cluster_features

# --- Sidebar (for navigation & upload) ---
st.sidebar.title("Athlete360 🏀")
uploaded_file = st.sidebar.file_uploader("Upload Player Game Log (CSV)", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

# --- Main Application ---
try:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File Uploaded Successfully!")
    
    # Run feature engineering
    # processed_data, perf_cols, injury_cols, cluster_cols = feature_engineer(data)
    
    # NOTE: For this example, we'll assume the uploaded CSV has all features
    # and just use the column names from your notebook.
    processed_data = data.copy()
    
    # Manually define feature lists (as from your notebook)
    perf_cols = [
        'minutes_played', 'days_since_last_game', 'back_to_back_games', 'games_last_7', 'games_last_30',
        'rolling_points_mean', 'rolling_points_std', 'rolling_assists_mean', 'rolling_assists_std',
        'rolling_rebounds_mean', 'rolling_rebounds_std', 'rolling_field_goal_pct_mean', 'rolling_field_goal_pct_std',
        'rolling_3pt_pct_mean', 'rolling_3pt_pct_std', 'rolling_free_throws_mean', 'rolling_free_throws_std',
        'rolling_turnovers_mean', 'rolling_turnovers_std', 'rolling_minutes_mean', 'rolling_minutes_std',
        'last_vs_rolling_points_diff', 'hot_streak_flag', 'hot_streak_count', 'fatigue_index', 'gap_days',
        'opponent_avg_blocks', 'opponent_avg_steals', 'opponent_def_rating', 'opponent_pace', 
        'opponent_avg_points_allowed', 'opponent_field_goals_pct_allowed', 'opponent_3pt_pct_allowed',
        'opponent_points_paint_allowed'
    ]
    
    injury_cols = [
        'minutes_played', 'points', 'assists', 'total_rebounds', 'turnovers', 'fatigue_index', 
        'days_since_last_game', 'rolling_minutes_mean', 'rolling_points_mean', 
        'opponent_def_rating', 'opponent_avg_points_allowed'
    ]
    
    cluster_cols = [
        'rolling_points_mean', 'rolling_points_std', 'rolling_assists_mean', 'rolling_assists_std',
        'rolling_rebounds_mean', 'rolling_rebounds_std', 'rolling_field_goal_pct_mean', 
        'rolling_3pt_pct_mean', 'rolling_free_throws_mean', 'minutes_played', 'fatigue_index'
    ]

    # --- Data Filtering (Ensure all columns are present) ---
    # Drop rows where essential model inputs are missing
    processed_data = processed_data.dropna(subset=perf_cols + injury_cols + cluster_cols)
    
    if processed_data.empty:
        st.error("Uploaded CSV is missing required columns or has no valid data after cleaning. Please check the file.")
        st.stop()

    # --- Run All Models ---
    
    # Q1: Performance Prediction
    X_perf = processed_data[perf_cols]
    processed_data['predicted_points'] = lgbm_model.predict(X_perf)
    
    # Q2: Injury Risk Prediction
    X_injury = processed_data[injury_cols]
    processed_data['injury_risk_proba'] = rf_injury_model.predict_proba(X_injury)[:, 1] # Get probability of class 1
    processed_data['injury_risk_flag'] = rf_injury_model.predict(X_injury)
    
    # Q3: Training Clusters
    X_cluster = processed_data[cluster_cols]
    X_cluster_scaled = cluster_scaler.transform(X_cluster)
    processed_data['cluster'] = kmeans_model.predict(X_cluster_scaled)
    
    # Map cluster recommendations (from your notebook)
    cluster_map = {
        0: "Focus: High-Scoring Playmaker", # Adjust names based on your radar plot
        1: "Focus: Balanced Contributor",
        2: "Focus: Low-Output / Fatigue-Prone",
        3: "Focus: High-Usage Scorer"
    }
    processed_data['training_focus'] = processed_data['cluster'].map(cluster_map).fillna("General")

    # Q4: Athlete Intelligence Index (AII)
    # (Using the logic from your notebook)
    w_perf = 0.40
    w_fatigue = 0.25
    w_injury = 0.25
    
    # Normalize inputs
    norm_perf = processed_data['predicted_points'] / processed_data['predicted_points'].max()
    norm_fatigue = 1 - (processed_data['fatigue_index'] / processed_data['fatigue_index'].max()) # Invert: high fatigue = low score
    norm_injury = 1 - processed_data['injury_risk_proba'] # Invert: high risk = low score
    
    processed_data['AII_Score'] = (w_perf * norm_perf + w_fatigue * norm_fatigue + w_injury * norm_injury) * 100
    
    # --- UI Tabs ---
    st.title("Athlete360 Performance Dashboard")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 AII Dashboard", "📈 Performance Prediction", "❤️ Injury Risk", "🏃‍♂️ Training Clusters"])

    with tab1:
        st.header("Athlete Intelligence Index (AII)")
        st.markdown("A unified score (0-100) combining predicted performance, fatigue, and injury risk.")
        
        # Select player for detail view
        player_list = processed_data['player_name'].unique()
        selected_player = st.selectbox("Select Player to Analyze", player_list)
        
        player_data = processed_data[processed_data['player_name'] == selected_player].iloc[-1] # Get latest game
        
        col1, col2, col3 = st.columns(3)
        col1.metric("AII Score", f"{player_data['AII_Score']:.1f}")
        col2.metric("Predicted Points (Next Game)", f"{player_data['predicted_points']:.1f}")
        col3.metric("Injury Risk", f"{player_data['injury_risk_proba']*100:.1f}%", 
                     delta=f"{player_data['injury_risk_proba']*100 - 20:.1f}% vs Avg", delta_color="inverse")
        
        st.subheader("Top 5 Players by AII")
        st.dataframe(processed_data.sort_values('AII_Score', ascending=False).head(5)[
            ['player_name', 'AII_Score', 'predicted_points', 'injury_risk_proba', 'training_focus']
        ])

    with tab2:
        st.header("Q1: Next-Game Performance Prediction")
        fig = px.scatter(processed_data, x="rolling_points_mean", y="predicted_points", 
                         color="fatigue_index", title="Predicted Points vs. Recent Form")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Q2: Injury & Fatigue Risk")
        risky_players = processed_data[processed_data['injury_risk_proba'] > 0.5].sort_values('injury_risk_proba', ascending=False)
        st.subheader(f"Players at High Risk: {len(risky_players)}")
        st.dataframe(risky_players[['player_name', 'injury_risk_proba', 'fatigue_index', 'minutes_played']])
    
    with tab4:
        st.header("Q3: Player Training Clusters")
        st.markdown("Players clustered by performance style and fatigue to generate recommendations.")
        fig = px.scatter(processed_data.sample(min(1000, len(processed_data))), 
                         x="rolling_points_mean", y="rolling_assists_mean", 
                         color="training_focus", size="fatigue_index",
                         title="Player Clusters")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while processing the file: {e}")
    st.exception(e)