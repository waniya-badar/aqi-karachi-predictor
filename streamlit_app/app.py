"""
Streamlit Dashboard - AQI Predictor for Karachi
Interactive web application for viewing AQI predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
import json

from src.mongodb_handler import MongoDBHandler
from src.data_fetcher import AQICNFetcher
from src.feature_engineering import FeatureEngineer


# Page configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name='random_forest'):
    """Load trained model from disk"""
    try:
        model_path = f'models/saved_models/{model_name}_latest.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load feature scaler"""
    try:
        scaler_path = 'models/saved_models/scaler_latest.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None


@st.cache_resource
def load_feature_names():
    """Load feature names"""
    try:
        with open('models/saved_models/feature_names.json', 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return []


def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "#00e400", "😊"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "😷"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "😨"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "🤢"
    else:
        return "Hazardous", "#7e0023", "HAZARDOUS"


def create_aqi_gauge(aqi_value):
    """Create AQI gauge chart"""
    category, color, emoji = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current AQI {emoji}", 'font': {'size': 24}},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#00e400'},
                {'range': [50, 100], 'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig, category


def predict_future_aqi(model, scaler, current_features, feature_names, days=3):
    """Predict AQI for next N days with proper feature handling"""
    predictions = []
    
    try:
        # Create initial feature dataframe with proper dtypes
        features_df = pd.DataFrame([current_features])
        
        # For each day to predict
        for day in range(1, days + 1):
            try:
                # Ensure all required features exist
                missing_features = [f for f in feature_names if f not in features_df.columns]
                
                if missing_features:
                    # Create missing lag features by copying current values
                    for f in missing_features:
                        if 'lag' in f or 'rolling' in f:
                            # Use current AQI for lag features
                            base_col = f.split('_')[0]  # e.g., 'aqi' from 'aqi_lag_1h'
                            if base_col in features_df.columns:
                                features_df[f] = float(features_df[base_col].iloc[0])
                            else:
                                features_df[f] = float(current_features.get('aqi', 100))
                        else:
                            # For other features, use default value
                            features_df[f] = float(current_features.get(f, 0))
                
                # Select only required features in correct order
                X = features_df[feature_names].copy().iloc[-1:]
                
                # Ensure dataframe has feature names
                X.columns = feature_names
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Predict
                pred = model.predict(X_scaled)[0]
                aqi_value = max(0, float(pred))
                
                predictions.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'aqi': aqi_value
                })
                
                # Update features for next day prediction
                # Shift temporal features
                if 'hour' in features_df.columns:
                    features_df.loc[0, 'hour'] = float((features_df.loc[0, 'hour'] + 24) % 24)
                if 'day' in features_df.columns:
                    features_df.loc[0, 'day'] = float((features_df.loc[0, 'day'] + 1) % 31)
                if 'month' in features_df.columns:
                    features_df.loc[0, 'month'] = float((features_df.loc[0, 'month'] % 12) + 1)
                
                # Update lag features with predicted value
                for col in features_df.columns:
                    if 'lag' in col or 'rolling' in col:
                        features_df.loc[0, col] = aqi_value
                        
            except Exception as e:
                st.warning(f"Error predicting day {day}: {str(e)}")
                predictions.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'aqi': float(current_features.get('aqi', 100))
                })
        
        return predictions
        
    except Exception as e:
        st.error(f"Error in predict_future_aqi: {str(e)}")
        # Return fallback predictions
        base_aqi = float(current_features.get('aqi', 100))
        return [
            {'day': i, 'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'), 'aqi': base_aqi}
            for i in range(1, days + 1)
        ]


def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">Karachi AQI Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Air Quality Index Prediction for the Next 3 Days")
    
    st.sidebar.title("SETTINGS & MODEL SELECTION")
    
    st.sidebar.markdown("### Select Prediction Model")
    st.sidebar.markdown("**Ridge Regression is the BEST MODEL (Default)**")
    
    # Model options with performance indicators
    model_options = {
        "Ridge Regression (BEST - R²=0.9947)": "ridge",
        "Gradient Boosting (R²=0.9548)": "gradient_boosting",
        "Random Forest (R²=0.9059)": "random_forest"
    }
    
    selected_model_display = st.sidebar.selectbox(
        "Choose a model for predictions:",
        list(model_options.keys()),
        index=0  # Ridge is default (index 0)
    )
    
    model_choice = model_options[selected_model_display]
    
    # Display model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### SELECTED MODEL DETAILS")
    
    model_details = {
        "ridge": {
            "name": "Ridge Regression",
            "train_r2": 0.9911,
            "test_r2": 0.9947,
            "rmse": 1.24,
            "mae": 1.01,
            "status": "BEST PERFORMER",
            "color": "green"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "train_r2": 0.9935,
            "test_r2": 0.9548,
            "rmse": 3.62,
            "mae": 2.87,
            "status": "Strong Performance",
            "color": "blue"
        },
        "random_forest": {
            "name": "Random Forest",
            "train_r2": 0.9129,
            "test_r2": 0.9059,
            "rmse": 5.22,
            "mae": 5.06,
            "status": "Good Performance",
            "color": "orange"
        }
    }
    
    selected_info = model_details[model_choice]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Test R²", f"{selected_info['test_r2']:.4f}")
    with col2:
        st.metric("RMSE", f"{selected_info['rmse']:.2f}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Train R²", f"{selected_info['train_r2']:.4f}")
    with col2:
        st.metric("MAE", f"{selected_info['mae']:.2f}")
    
    st.sidebar.info(f"Status: {selected_info['status']}")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data updates hourly via automated pipeline")
    
    # Main content
    try:
        # Load model and data
        with st.spinner("Loading model and fetching data..."):
            model = load_model(model_choice)
            scaler = load_scaler()
            feature_names = load_feature_names()
            
            db_handler = MongoDBHandler()
            fetcher = AQICNFetcher()
            engineer = FeatureEngineer()
            
            # Get current data
            raw_data = fetcher.fetch_current_data()
            if not raw_data:
                st.error("Failed to fetch current AQI data from API")
                return
            
            # Create features
            current_features = engineer.create_features(raw_data)
            current_aqi = current_features['aqi']
        
        # Current AQI Section
        st.markdown("## CURRENT AIR QUALITY STATUS")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # AQI Gauge
            gauge_fig, category = create_aqi_gauge(current_aqi)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Category info
            st.markdown(f"### Status: **{category}**")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            # Key metrics
            st.markdown("### 🔬 Current Pollutant Levels")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("PM2.5", f"{current_features.get('pm25', 0):.1f} μg/m³")
                st.metric("PM10", f"{current_features.get('pm10', 0):.1f} μg/m³")
            
            with metric_col2:
                st.metric("O₃", f"{current_features.get('o3', 0):.1f} ppb")
                st.metric("NO₂", f"{current_features.get('no2', 0):.1f} ppb")
            
            with metric_col3:
                st.metric("Temperature", f"{current_features.get('temperature', 0):.1f}°C")
                st.metric("Humidity", f"{current_features.get('humidity', 0):.0f}%")
        
        # Health Advisory with Detailed Alerts
        if current_aqi > 300:
            st.error("""
                HAZARDOUS ALERT - EMERGENCY CONDITIONS
                - AQI: {} (Hazardous)
                - IMMEDIATE ACTION REQUIRED
                - Everyone should avoid ALL outdoor activities
                - Stay indoors with windows closed
                - Use air purifiers if available
                - Wear N95 masks if you must go outside
                - Seek medical attention if experiencing symptoms
                """.format(int(current_aqi)))
        elif current_aqi > 200:
            st.error("""
                VERY UNHEALTHY ALERT
                - AQI: {} (Very Unhealthy)
                - Health alert: Everyone may experience serious health effects
                - Children, elderly, and those with respiratory conditions should remain indoors
                - General population should greatly limit outdoor activities
                - Wear masks if outdoor exposure is unavoidable
                """.format(int(current_aqi)))
        elif current_aqi > 150:
            st.warning("""
                UNHEALTHY ALERT
                - AQI: {} (Unhealthy)
                - Everyone may begin to experience health effects
                - Sensitive groups should avoid outdoor activities
                - General population should limit prolonged outdoor exertion
                - Consider wearing masks during outdoor activities
                """.format(int(current_aqi)))
        elif current_aqi > 100:
            st.info("""
                MODERATE - Sensitive Groups Advisory
                - AQI: {} (Moderate)
                - Unusually sensitive people should consider limiting prolonged outdoor exertion
                - General population can enjoy normal outdoor activities
                - Monitor symptoms if you have respiratory conditions
                """.format(int(current_aqi)))
        else:
            st.success("""
                GOOD AIR QUALITY
                - AQI: {} (Good)
                - It's a great day to be outdoors!
                - No health impacts expected
                - Enjoy outdoor activities
                """.format(int(current_aqi)))
        
        st.markdown("---")
        
        # Model Comparison Section - Professional Graphs
        st.markdown("## MODEL PERFORMANCE COMPARISON")
        st.markdown("Results from 3 trained machine learning models on 2,880 samples")
        
        try:
            with st.spinner("Loading model comparison..."):
                # Load model registry
                registry_path = 'models/model_registry.json'
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            registry = json.loads(content)
                        else:
                            registry = []
                else:
                    registry = []
                
                if registry:
                    # Create comparison data
                    models_data = []
                    for entry in registry:
                        if 'model_name' in entry:
                            models_data.append({
                                'Model': entry['model_name'].replace('_', ' ').title(),
                                'Train R²': round(entry.get('train_r2', 0), 4),
                                'Test R²': round(entry.get('test_r2', 0), 4),
                                'Test RMSE': round(entry.get('test_rmse', 0), 2),
                                'Test MAE': round(entry.get('test_mae', 0), 2),
                            })
                    
                    if models_data:
                        models_df = pd.DataFrame(models_data)
                        
                        # Display metrics table
                        st.markdown("### Detailed Model Metrics Table")
                        st.dataframe(
                            models_df.set_index('Model'),
                            use_container_width=True,
                            column_config={
                                "Train R²": st.column_config.NumberColumn(format="%.4f"),
                                "Test R²": st.column_config.NumberColumn(format="%.4f"),
                                "Test RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "Test MAE": st.column_config.NumberColumn(format="%.2f"),
                            }
                        )
                        
                        st.markdown("---")
                        
                        # Model 1: Random Forest Details
                        st.markdown("### 1. Random Forest Model")
                        col1, col2, col3, col4 = st.columns(4)
                        rf_data = models_df[models_df['Model'] == 'Random Forest'].iloc[0]
                        with col1:
                            st.metric("Train R²", f"{rf_data['Train R²']:.4f}")
                        with col2:
                            st.metric("Test R²", f"{rf_data['Test R²']:.4f}")
                        with col3:
                            st.metric("Test RMSE", f"{rf_data['Test RMSE']:.2f}")
                        with col4:
                            st.metric("Test MAE", f"{rf_data['Test MAE']:.2f}")
                        
                        st.markdown("**Model Description:** Ensemble of 100 independent decision trees. Each tree learns patterns independently, and final predictions are averaged across all trees.")
                        st.markdown("**Performance:** Good generalization with reasonable error metrics.")
                        
                        st.markdown("---")
                        
                        # Model 2: Gradient Boosting Details
                        st.markdown("### 2. Gradient Boosting Model")
                        col1, col2, col3, col4 = st.columns(4)
                        gb_data = models_df[models_df['Model'] == 'Gradient Boosting'].iloc[0]
                        with col1:
                            st.metric("Train R²", f"{gb_data['Train R²']:.4f}")
                        with col2:
                            st.metric("Test R²", f"{gb_data['Test R²']:.4f}")
                        with col3:
                            st.metric("Test RMSE", f"{gb_data['Test RMSE']:.2f}")
                        with col4:
                            st.metric("Test MAE", f"{gb_data['Test MAE']:.2f}")
                        
                        st.markdown("**Model Description:** Sequential ensemble learning where each tree corrects errors from previous trees. Iterative refinement process creates stronger predictions.")
                        st.markdown("**Performance:** Strong performance with better accuracy than Random Forest.")
                        
                        st.markdown("---")
                        
                        # Model 3: Ridge Regression Details (Best Model)
                        st.markdown("### 3. Ridge Regression Model [BEST PERFORMING]")
                        col1, col2, col3, col4 = st.columns(4)
                        ridge_data = models_df[models_df['Model'] == 'Ridge Regression'].iloc[0]
                        with col1:
                            st.metric("Train R²", f"{ridge_data['Train R²']:.4f}", delta="Highest")
                        with col2:
                            st.metric("Test R²", f"{ridge_data['Test R²']:.4f}", delta="+0.0399")
                        with col3:
                            st.metric("Test RMSE", f"{ridge_data['Test RMSE']:.2f}", delta="Lowest")
                        with col4:
                            st.metric("Test MAE", f"{ridge_data['Test MAE']:.2f}", delta="Lowest")
                        
                        st.markdown("**Model Description:** Linear regression with L2 regularization. Uses single mathematical equation: AQI = w1*x1 + w2*x2 + ... + w46*x46 + bias. Regularization prevents overfitting.")
                        st.markdown("**Performance:** BEST OVERALL - Explains 99.47% of variance with lowest error margins. Selected for production use.")
                        
                        st.success("RIDGE REGRESSION SELECTED FOR AQI FORECASTING")
                        
                        st.markdown("---")
                        st.markdown("## COMPARATIVE ANALYSIS CHARTS")
                        
                        # R² Score Comparison - Enhanced
                        st.markdown("### R² Score Comparison (Train vs Test)")
                        st.markdown("R² ranges from 0 to 1. Higher is better. Values above 0.9 indicate excellent fit.")
                        
                        fig_r2 = go.Figure()
                        
                        fig_r2.add_trace(go.Bar(
                            x=models_df['Model'],
                            y=models_df['Train R²'],
                            name='Train R²',
                            marker_color='rgb(31, 119, 180)',
                            text=models_df['Train R²'].round(4),
                            textposition='outside',
                        ))
                        
                        fig_r2.add_trace(go.Bar(
                            x=models_df['Model'],
                            y=models_df['Test R²'],
                            name='Test R²',
                            marker_color=['rgb(255, 127, 14)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)'],
                            text=models_df['Test R²'].round(4),
                            textposition='outside',
                        ))
                        
                        fig_r2.update_layout(
                            barmode='group',
                            title='R² Score: Variance Explained by Each Model',
                            xaxis_title='Model',
                            yaxis_title='R² Score',
                            height=400,
                            hovermode='x unified',
                            template='plotly_white',
                            showlegend=True,
                            yaxis=dict(range=[0, 1.05])
                        )
                        
                        st.plotly_chart(fig_r2, use_container_width=True)
                        
                        # Error Metrics Comparison - Enhanced
                        st.markdown("### Error Metrics Comparison (RMSE and MAE)")
                        st.markdown("RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) measure prediction error. Lower values indicate better performance.")
                        
                        fig_error = go.Figure()
                        
                        fig_error.add_trace(go.Bar(
                            x=models_df['Model'],
                            y=models_df['Test RMSE'],
                            name='RMSE',
                            marker_color='rgb(214, 39, 40)',
                            text=models_df['Test RMSE'].round(2),
                            textposition='outside',
                        ))
                        
                        fig_error.add_trace(go.Bar(
                            x=models_df['Model'],
                            y=models_df['Test MAE'],
                            name='MAE',
                            marker_color=['rgb(227, 119, 194)', 'rgb(227, 119, 194)', 'rgb(44, 160, 44)'],
                            text=models_df['Test MAE'].round(2),
                            textposition='outside',
                        ))
                        
                        fig_error.update_layout(
                            barmode='group',
                            title='Error Metrics: Average Prediction Deviation',
                            xaxis_title='Model',
                            yaxis_title='Error Value (AQI Points)',
                            height=400,
                            hovermode='x unified',
                            template='plotly_white',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_error, use_container_width=True)
                        
                        # Radar Chart - Model Comparison
                        st.markdown("### Overall Model Performance Radar Chart")
                        
                        # Normalize metrics to 0-1 scale for radar chart
                        radar_data = models_df.copy()
                        radar_data['Train R² Norm'] = radar_data['Train R²']
                        radar_data['Test R² Norm'] = radar_data['Test R²']
                        radar_data['RMSE Norm'] = 1 - (radar_data['Test RMSE'] / radar_data['Test RMSE'].max())
                        radar_data['MAE Norm'] = 1 - (radar_data['Test MAE'] / radar_data['Test MAE'].max())
                        
                        fig_radar = go.Figure()
                        
                        for idx, row in radar_data.iterrows():
                            fig_radar.add_trace(go.Scatterpolar(
                                r=[row['Train R² Norm'], row['Test R² Norm'], row['RMSE Norm'], row['MAE Norm']],
                                theta=['Train R²', 'Test R²', 'Low RMSE', 'Low MAE'],
                                fill='toself',
                                name=row['Model'],
                                line_color=['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)'][idx],
                            ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            height=500,
                            title='Overall Performance Comparison (Normalized Scale)',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                        
                        # Summary Statistics
                        st.markdown("---")
                        st.markdown("## PERFORMANCE SUMMARY")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### Best R² Score")
                            st.markdown(f"**{ridge_data['Test R²']:.4f}** (Ridge Regression)")
                            st.markdown("99.47% of AQI variance explained")
                        
                        with col2:
                            st.markdown("### Lowest RMSE")
                            st.markdown(f"**{ridge_data['Test RMSE']:.2f}** (Ridge Regression)")
                            st.markdown("Approximately 1.24 AQI points average error")
                        
                        with col3:
                            st.markdown("### Lowest MAE")
                            st.markdown(f"**{ridge_data['Test MAE']:.2f}** (Ridge Regression)")
                            st.markdown("Approximately 1.01 AQI points average deviation")
                        
                        st.markdown("---")
                        st.info(
                            "CONCLUSION: Ridge Regression provides the best balance of accuracy, "
                            "generalization, and simplicity. Selected for production AQI forecasting."
                        )
                        
                        # COMPREHENSIVE ALL MODELS GRAPH SECTION
                        st.markdown("---")
                        st.markdown("## ALL MODELS COMPREHENSIVE COMPARISON GRAPH")
                        st.markdown("Interactive visualization comparing all 3 models across key performance metrics")
                        
                        # Create comprehensive comparison figure
                        fig_comprehensive = go.Figure()
                        
                        # Add metrics for each model
                        models_list = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']
                        colors = ['rgb(255, 127, 14)', 'rgb(214, 39, 40)', 'rgb(44, 160, 44)']  # Orange, Red, Green
                        
                        # Test R² Score
                        fig_comprehensive.add_trace(go.Bar(
                            x=models_list,
                            y=[0.9059, 0.9548, 0.9947],
                            name='Test R² Score',
                            marker_color=colors,
                            text=['0.9059', '0.9548', '0.9947'],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Test R²: %{y:.4f}<extra></extra>'
                        ))
                        
                        fig_comprehensive.update_layout(
                            title='ALL 3 MODELS - TEST R² SCORE COMPARISON (Higher is Better)',
                            xaxis_title='Machine Learning Model',
                            yaxis_title='Test R² Score',
                            height=500,
                            template='plotly_white',
                            showlegend=False,
                            yaxis=dict(range=[0, 1.05]),
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig_comprehensive, use_container_width=True)
                        
                        # Create error comparison comprehensive figure
                        fig_error_comprehensive = go.Figure()
                        
                        fig_error_comprehensive.add_trace(go.Bar(
                            x=models_list,
                            y=[5.22, 3.62, 1.24],
                            name='RMSE',
                            marker_color=['rgb(255, 127, 14)', 'rgb(214, 39, 40)', 'rgb(44, 160, 44)'],
                            text=['5.22', '3.62', '1.24'],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f}<extra></extra>'
                        ))
                        
                        fig_error_comprehensive.add_trace(go.Bar(
                            x=models_list,
                            y=[5.06, 2.87, 1.01],
                            name='MAE',
                            marker_color=['rgb(255, 180, 124)', 'rgb(240, 120, 100)', 'rgb(120, 220, 120)'],
                            text=['5.06', '2.87', '1.01'],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:.2f}<extra></extra>'
                        ))
                        
                        fig_error_comprehensive.update_layout(
                            barmode='group',
                            title='ALL 3 MODELS - ERROR METRICS COMPARISON (Lower is Better)',
                            xaxis_title='Machine Learning Model',
                            yaxis_title='Error Value (AQI Points)',
                            height=500,
                            template='plotly_white',
                            hovermode='x unified',
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig_error_comprehensive, use_container_width=True)
                        
                        # Comprehensive metrics table for all models
                        st.markdown("### Complete Metrics Comparison Table")
                        
                        all_models_df = pd.DataFrame({
                            'Model': ['Random Forest', 'Gradient Boosting', 'Ridge Regression'],
                            'Train R²': [0.9129, 0.9935, 0.9911],
                            'Test R² ': [0.9059, 0.9548, 0.9947],
                            'Test RMSE': [5.22, 3.62, 1.24],
                            'Test MAE': [5.06, 2.87, 1.01],
                            'Status': ['Good', 'Strong', 'BEST']
                        })
                        
                        st.dataframe(
                            all_models_df.set_index('Model'),
                            use_container_width=True,
                            column_config={
                                "Train R²": st.column_config.NumberColumn(format="%.4f"),
                                "Test R² ": st.column_config.NumberColumn(format="%.4f"),
                                "Test RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "Test MAE": st.column_config.NumberColumn(format="%.2f"),
                            }
                        )
                        
                        # Winner announcement
                        st.markdown("---")
                        st.success("WINNER: Ridge Regression (R²=0.9947, RMSE=1.24) - Selected by Default in Sidebar")
        
        except Exception as e:
            st.warning(f"Could not load model comparison: {str(e)}")
        
        st.markdown("---")
        
        # Predictions Section
        st.markdown("## 3-DAY AQI FORECAST PREDICTION")
        
        if model and scaler and feature_names:
            with st.spinner("Generating 3-day predictions..."):
                predictions = predict_future_aqi(model, scaler, current_features, feature_names, days=3)
            
            if predictions:
                # Display predictions in cards
                pred_cols = st.columns(3)
                
                for i, pred in enumerate(predictions):
                    with pred_cols[i]:
                        category, color, emoji = get_aqi_category(pred['aqi'])
                        
                        st.markdown(f"""
                            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #333;">
                                <h3 style="margin: 0; color: black;">Day {pred['day']}</h3>
                                <p style="margin: 5px 0; color: black; font-size: 12px;">{pred['date']}</p>
                                <h1 style="margin: 10px 0; color: black;">{pred['aqi']:.0f}</h1>
                                <h2 style="margin: 5px 0;">{emoji}</h2>
                                <p style="margin: 0; color: black; font-weight: bold;">{category}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Comprehensive trend chart with current + forecast
                st.markdown("### AQI 3-DAY FORECAST TREND CHART")
                
                trend_data = pd.DataFrame(predictions)
                trend_data['date'] = pd.to_datetime(trend_data['date'])
                
                # Add current AQI
                current_date = pd.DataFrame([{'date': datetime.now(), 'aqi': current_aqi, 'type': 'Current'}])
                trend_data['type'] = 'Forecast'
                
                combined_data = pd.concat([current_date, trend_data], ignore_index=True)
                
                fig_forecast = go.Figure()
                
                # Current AQI point
                fig_forecast.add_trace(go.Scatter(
                    x=combined_data[combined_data['type'] == 'Current']['date'],
                    y=combined_data[combined_data['type'] == 'Current']['aqi'],
                    mode='markers',
                    marker=dict(size=15, color='#1f77b4', symbol='diamond'),
                    name='Current AQI',
                    text=['Current'],
                    hovertemplate='<b>Current AQI</b><br>%{y:.0f}<extra></extra>'
                ))
                
                # Forecast line
                fig_forecast.add_trace(go.Scatter(
                    x=combined_data[combined_data['type'] == 'Forecast']['date'],
                    y=combined_data[combined_data['type'] == 'Forecast']['aqi'],
                    mode='lines+markers',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=12),
                    name='3-Day Forecast',
                    hovertemplate='<b>Day %{x|%a}</b><br>Predicted AQI: %{y:.0f}<extra></extra>'
                ))
                
                # Add connecting line
                fig_forecast.add_trace(go.Scatter(
                    x=combined_data['date'],
                    y=combined_data['aqi'],
                    mode='lines',
                    line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                    name='Trend',
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Add AQI category bands
                fig_forecast.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.1, annotation_text="Good", annotation_position="left")
                fig_forecast.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.1, annotation_text="Moderate", annotation_position="left")
                fig_forecast.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1, annotation_text="Unhealthy SG", annotation_position="left")
                fig_forecast.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1, annotation_text="Unhealthy", annotation_position="left")
                fig_forecast.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.1, annotation_text="Very Unhealthy", annotation_position="left")
                fig_forecast.add_hrect(y0=300, y1=500, fillcolor="#7e0023", opacity=0.1, annotation_text="Hazardous", annotation_position="left")
                
                fig_forecast.update_layout(
                    title='Karachi AQI: Current Status & 3-Day Forecast',
                    xaxis_title='Date',
                    yaxis_title='AQI Value',
                    height=450,
                    hovermode='x unified',
                    legend=dict(x=0, y=1),
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast summary
                st.markdown("### FORECAST SUMMARY")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                avg_aqi = trend_data['aqi'].mean()
                max_aqi = trend_data['aqi'].max()
                min_aqi = trend_data['aqi'].min()
                
                with summary_col1:
                    st.metric("Average AQI (3 days)", f"{avg_aqi:.0f}")
                
                with summary_col2:
                    st.metric("Highest AQI", f"{max_aqi:.0f}")
                    _, max_cat, _ = get_aqi_category(max_aqi)
                    st.caption(f"Category: {max_cat}")
                
                with summary_col3:
                    st.metric("Lowest AQI", f"{min_aqi:.0f}")
                    _, min_cat, _ = get_aqi_category(min_aqi)
                    st.caption(f"Category: {min_cat}")
                
                # Forecast recommendations
                st.markdown("### 3-DAY FORECAST RECOMMENDATIONS")
                
                if avg_aqi > 200:
                    st.error("Expect hazardous air quality over the next 3 days. Plan indoor activities and use air purifiers.")
                elif avg_aqi > 150:
                    st.warning("Expect unhealthy air quality. Sensitive groups should limit outdoor activities.")
                elif avg_aqi > 100:
                    st.info("Expect moderate air quality. Consider wearing masks for outdoor activities.")
                else:
                    st.success("Good air quality expected. Safe to plan outdoor activities!")
        
        else:
            st.error("Unable to load model or features for prediction")

        
        st.markdown("---")
        
        # Historical Data Section
        st.markdown("## HISTORICAL AQI DATA - REAL-TIME READINGS")
        
        # Day range selector
        hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
        
        with hist_col1:
            if st.button("Last 3 Days", use_container_width=True):
                st.session_state.history_days = 3
        with hist_col2:
            if st.button("Last 7 Days", use_container_width=True):
                st.session_state.history_days = 7
        with hist_col3:
            if st.button("Last 30 Days", use_container_width=True):
                st.session_state.history_days = 30
        with hist_col4:
            if st.button("Last 90 Days", use_container_width=True):
                st.session_state.history_days = 90
        
        # Initialize session state
        if 'history_days' not in st.session_state:
            st.session_state.history_days = 7
        
        history_days = st.session_state.history_days
        
        with st.spinner(f"Loading historical data for last {history_days} days..."):
            # Fetch data for specified number of days
            hist_df = db_handler.get_training_data(days=history_days)
            
            if hist_df is not None and len(hist_df) > 0:
                # Ensure dataframe has required columns
                if 'timestamp' in hist_df.columns and 'aqi' in hist_df.columns:
                    # Convert timestamp to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(hist_df['timestamp']):
                        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                    
                    # Sort by timestamp
                    hist_df = hist_df.sort_values('timestamp')
                    
                    # Ensure AQI is numeric
                    hist_df['aqi'] = pd.to_numeric(hist_df['aqi'], errors='coerce')
                    
                    # Remove any NaN values
                    hist_df = hist_df.dropna(subset=['aqi'])
                    
                    # Display date range info
                    if len(hist_df) > 0:
                        date_range = f"{hist_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {hist_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
                        st.info(f"Data Range: {date_range} | Total Records: {len(hist_df)}")
                    
                    if len(hist_df) > 0:
                        # Create a more detailed plot with markers and values
                        hist_fig = go.Figure()
                        
                        # Add line trace with markers
                        hist_fig.add_trace(go.Scatter(
                            x=hist_df['timestamp'],
                            y=hist_df['aqi'],
                            mode='lines+markers',
                            name='AQI',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8, color='#1f77b4', symbol='circle'),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            hovertemplate='<b>Date:</b> %{x}<br><b>AQI:</b> %{y:.0f}<extra></extra>',
                            text=[f"AQI: {val:.0f}" for val in hist_df['aqi']],
                            textposition='top center'
                        ))
                        
                        # Update layout with better formatting
                        hist_fig.update_layout(
                            title=f'AQI Trend - Last {history_days} Days (Real-Time Readings)',
                            xaxis_title='Date & Time',
                            yaxis_title='AQI Value',
                            height=500,
                            hovermode='x unified',
                            template='plotly_white',
                            xaxis=dict(
                                tickformat='%b %d, %H:%M',
                                tickangle=-45
                            ),
                            yaxis=dict(
                                gridcolor='rgba(200,200,200,0.3)',
                                zeroline=False
                            )
                        )
                        
                        st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # Display the data table
                        st.markdown(f"### Data Table - Last {history_days} Days (Latest 50 Records)")
                        display_df = hist_df.copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        display_df = display_df.rename(columns={'timestamp': 'Date & Time', 'aqi': 'AQI Value'})
                        
                        st.dataframe(
                            display_df[['Date & Time', 'AQI Value']].tail(50),
                            use_container_width=True,
                            column_config={
                                "AQI Value": st.column_config.NumberColumn(format="%d"),
                            }
                        )
                        
                        # Statistics
                        st.markdown(f"### STATISTICS - LAST {history_days} DAYS")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.metric("Average AQI", f"{hist_df['aqi'].mean():.0f}")
                        with stat_col2:
                            st.metric("Max AQI", f"{hist_df['aqi'].max():.0f}")
                        with stat_col3:
                            st.metric("Min AQI", f"{hist_df['aqi'].min():.0f}")
                        with stat_col4:
                            st.metric("Std Dev", f"{hist_df['aqi'].std():.1f}")
                        
                        # Additional statistics
                        st.markdown("### DETAILED ANALYSIS")
                        analysis_col1, analysis_col2 = st.columns(2)
                        
                        with analysis_col1:
                            st.info(f"Total Records: {len(hist_df)}")
                        with analysis_col2:
                            median_aqi = hist_df['aqi'].median()
                            st.info(f"Median AQI: {median_aqi:.0f}")
                    else:
                        st.warning("No valid AQI data available for display")
                else:
                    st.warning("Historical data missing required columns (timestamp, aqi)")
            else:
                st.info("Historical data will appear here once collected. Collecting hourly AQI readings from MongoDB...")
        
        # Close DB connection
        db_handler.close()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>Data source: AQICN | Updated hourly | Made with Streamlit</p>
            <p>Predictions are estimates and should not be used as the sole basis for health decisions</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()