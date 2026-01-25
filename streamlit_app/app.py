"""
Streamlit Dashboard - AQI Predictor for Karachi
Interactive web application for viewing AQI predictions
Models are loaded from MongoDB (cloud-based serverless)
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=3600)
def get_db_handler():
    """Get MongoDB handler (cached)"""
    return MongoDBHandler()


@st.cache_resource(ttl=3600)
def load_model_from_mongodb(model_name: str):
    """Load trained model from MongoDB cloud storage"""
    try:
        db = get_db_handler()
        model_doc = db.get_model(model_name)
        
        if model_doc:
            model = pickle.loads(model_doc['model_binary'])
            scaler = pickle.loads(model_doc['scaler_binary'])
            feature_names = model_doc['feature_names']
            metrics = model_doc['metrics']
            return model, scaler, feature_names, metrics
        
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


@st.cache_data(ttl=3600)
def get_all_models_info():
    """Get metadata for all models from MongoDB"""
    try:
        db = get_db_handler()
        models = db.get_all_models_metadata()
        return models
    except Exception as e:
        return []


def get_predictions_from_mongodb():
    """Get latest predictions from MongoDB cloud - includes all models (NO CACHE - Always fresh)"""
    try:
        db = get_db_handler()
        # Sort by saved_at to get the most recently saved prediction
        predictions_doc = db.db.predictions.find_one(sort=[('saved_at', -1)])
        if predictions_doc:
            return predictions_doc
        return None
    except Exception as e:
        return None


def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


def create_aqi_gauge(aqi_value):
    """Create AQI gauge chart"""
    category, color = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        number={'font': {'size': 48, 'color': 'black'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current AQI: {int(aqi_value)}", 'font': {'size': 20}},
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
    """Predict AQI for next N days"""
    predictions = []
    current_aqi = float(current_features.get('aqi', 100))
    
    try:
        features = {}
        now = datetime.now()
        
        # Initialize features
        features['hour'] = now.hour
        features['day'] = now.day
        features['month'] = now.month
        features['year'] = now.year
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        features['time_of_day'] = 2
        
        # Weather features
        features['temperature'] = float(current_features.get('temperature', 25))
        features['humidity'] = float(current_features.get('humidity', 50))
        features['pressure'] = float(current_features.get('pressure', 1013))
        features['wind_speed'] = float(current_features.get('wind_speed', 10))
        features['visibility'] = float(current_features.get('visibility', 10))
        
        # Lag features
        for lag in [1, 3, 6, 12, 24]:
            features[f'aqi_lag_{lag}h'] = current_aqi
        
        features['aqi_rolling_mean_6h'] = current_aqi
        features['aqi_rolling_mean_12h'] = current_aqi
        features['aqi_rolling_mean_24h'] = current_aqi
        features['aqi_rolling_std_24h'] = current_aqi * 0.1
        features['aqi_change_1h'] = 0
        features['aqi_change_6h'] = 0
        
        predicted_values = [current_aqi]
        
        for day in range(1, days + 1):
            future_date = now + timedelta(days=day)
            features['hour'] = 12
            features['day'] = future_date.day
            features['month'] = future_date.month
            features['day_of_week'] = future_date.weekday()
            features['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
            
            # Small weather variations
            features['temperature'] += np.random.uniform(-2, 2)
            features['humidity'] = max(10, min(100, features['humidity'] + np.random.uniform(-5, 5)))
            
            X_values = [features.get(fname, 0) for fname in feature_names]
            X = np.array([X_values])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            
            aqi_value = max(0, min(500, float(pred)))
            if features['is_weekend']:
                aqi_value *= 0.95
            
            predictions.append({
                'day': day,
                'date': future_date.strftime('%Y-%m-%d'),
                'aqi': round(aqi_value, 1)
            })
            
            predicted_values.append(aqi_value)
            
            # Update lag features
            features['aqi_lag_24h'] = features['aqi_lag_12h']
            features['aqi_lag_12h'] = features['aqi_lag_6h']
            features['aqi_lag_6h'] = features['aqi_lag_3h']
            features['aqi_lag_3h'] = features['aqi_lag_1h']
            features['aqi_lag_1h'] = aqi_value
            
            recent = predicted_values[-6:] if len(predicted_values) >= 6 else predicted_values
            features['aqi_rolling_mean_6h'] = np.mean(recent)
            features['aqi_rolling_mean_12h'] = np.mean(predicted_values[-12:] if len(predicted_values) >= 12 else predicted_values)
            features['aqi_rolling_mean_24h'] = np.mean(predicted_values)
            features['aqi_rolling_std_24h'] = np.std(predicted_values) if len(predicted_values) > 1 else aqi_value * 0.1
            features['aqi_change_1h'] = aqi_value - predicted_values[-2] if len(predicted_values) >= 2 else 0
            features['aqi_change_6h'] = aqi_value - predicted_values[-6] if len(predicted_values) >= 6 else 0
        
        return predictions
        
    except Exception as e:
        base_aqi = current_aqi
        return [
            {'day': i, 'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'), 
             'aqi': round(base_aqi * (0.9 + i * 0.03), 1)}
            for i in range(1, days + 1)
        ]


def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">Karachi AQI Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Air Quality Prediction")
    
    # Load models info
    all_models = get_all_models_info()
    
    st.sidebar.title("Settings")
    
    if not all_models:
        st.sidebar.error("No models found in MongoDB!")
        st.stop()
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    
    model_options = {}
    model_details = {}
    best_model_name = None
    
    for model_info in all_models:
        model_key = model_info['model_name']
        metrics = model_info.get('metrics', {})
        display_name = model_key.replace('_', ' ').title()
        test_r2 = metrics.get('test_r2', 0)
        is_best = model_info.get('is_best', False)
        
        if is_best:
            best_model_name = model_key
            label = f"{display_name} (BEST - R2={test_r2:.4f})"
        else:
            label = f"{display_name} (R2={test_r2:.4f})"
        
        model_options[label] = model_key
        model_details[model_key] = {
            "name": display_name,
            "train_r2": metrics.get('train_r2', 0),
            "test_r2": test_r2,
            "rmse": metrics.get('test_rmse', 0),
            "mae": metrics.get('test_mae', 0),
            "is_best": is_best
        }
    
    # Sort options
    sorted_options = dict(sorted(model_options.items(), key=lambda x: 0 if x[1] == best_model_name else 1))
    
    selected_model_display = st.sidebar.selectbox("Choose model:", list(sorted_options.keys()), index=0)
    model_choice = sorted_options[selected_model_display]
    
    # Model metrics
    st.sidebar.markdown("### Model Metrics")
    info = model_details[model_choice]
    
    st.sidebar.metric("R\u00b2 Score", f"{info['test_r2']}")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("MAE", f"{info['mae']:.2f}")
    col2.metric("RMSE", f"{info['rmse']:.2f}")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    try:
        with st.spinner("Loading..."):
            model, scaler, feature_names, metrics = load_model_from_mongodb(model_choice)
            if model is None:
                st.error(f"Failed to load model '{model_choice}'")
                return
            db_handler = get_db_handler()
            # Current AQI
            st.markdown("## Current Air Quality")
            col1, col2 = st.columns([1, 2])
            # Use latest reading from feature store
            latest_features_df = db_handler.get_latest_features(limit=1)
            if latest_features_df is not None and len(latest_features_df) > 0:
                latest_features = latest_features_df.iloc[0]
                current_aqi = latest_features['aqi']
                category, color = get_aqi_category(current_aqi)
                with col1:
                    st.metric("Current AQI", current_aqi)
                    st.markdown(f"**Status: {category}**")
                with col2:
                    st.markdown("### Pollutant Levels")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("PM2.5", f"{latest_features.get('pm25', 0):.1f} ug/m3")
                    m1.metric("PM10", f"{latest_features.get('pm10', 0):.1f} ug/m3")
                    m2.metric("O3", f"{latest_features.get('o3', 0):.1f} ppb")
                    m2.metric("NO2", f"{latest_features.get('no2', 0):.1f} ppb")
                    # m3.metric("Temperature", f"{latest_features.get('temperature', 0):.1f} C")
                    m3.metric("Humidity", f"{latest_features.get('humidity', 0):.0f}%")
            else:
                st.warning("No current AQI data found in feature store.")
            
            # 3-Day Forecast
            st.markdown("## 3-Day AQI Forecast")
            
            # Load predictions from MongoDB
            mongo_predictions = get_predictions_from_mongodb()
            
            if mongo_predictions and 'all_model_predictions' in mongo_predictions:
                all_models_preds = mongo_predictions.get('all_model_predictions', [])
                st.info(f"Predictions available from {len(all_models_preds)} model(s) | Best model: {mongo_predictions.get('best_model', 'N/A')}")
                
                # Find predictions for selected model
                selected_model_preds = None
                for model_pred in all_models_preds:
                    if model_pred.get('model_name') == model_choice:
                        selected_model_preds = model_pred.get('predictions', [])
                        break
                
                # If selected model predictions found, use them
                if selected_model_preds:
                    predictions = selected_model_preds
                else:
                    # Fallback: use best model predictions or first available
                    predictions = all_models_preds[0].get('predictions', []) if all_models_preds else []
                
                if predictions:
                    pred_cols = st.columns(3)
                    for i, pred in enumerate(predictions[:3]):
                        with pred_cols[i]:
                            aqi_val = pred.get('aqi', 0)
                            category, color = get_aqi_category(aqi_val)
                            st.markdown(f"""
                                <div style='background-color: {color}; padding: 15px; border-radius: 8px; text-align: center;'>
                                    <h4 style='margin: 0; color: black;'>Day {pred.get('day', i+1)}</h4>
                                    <p style='margin: 5px 0; color: black;'>{pred.get('date', '')}</p>
                                    <h2 style='margin: 10px 0; color: black;'>{aqi_val:.1f}</h2>
                                    <p style='margin: 0; color: black;'>{category}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Forecast chart
                    st.markdown("### Forecast Trend")
                    trend_data = pd.DataFrame(predictions)
                    trend_data['date'] = pd.to_datetime(trend_data['date'])
                    current_aqi = mongo_predictions.get('current_aqi', 100)
                    current_df = pd.DataFrame([{'date': pd.to_datetime(mongo_predictions.get('saved_at')), 'aqi': current_aqi, 'type': 'Current'}])
                    trend_data['type'] = 'Forecast'
                    combined = pd.concat([current_df, trend_data], ignore_index=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=combined[combined['type'] == 'Current']['date'],
                        y=combined[combined['type'] == 'Current']['aqi'],
                        mode='markers',
                        marker=dict(size=12, color='#1f77b4', symbol='diamond'),
                        name='Current'
                    ))
                    fig.add_trace(go.Scatter(
                        x=combined[combined['type'] == 'Forecast']['date'],
                        y=combined[combined['type'] == 'Forecast']['aqi'],
                        mode='lines+markers',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        marker=dict(size=10),
                        name='Forecast'
                    ))
                    fig.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.1)
                    fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.1)
                    fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1)
                    fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1)
                    fig.update_layout(
                        title=f'AQI Forecast ({model_details[model_choice]["name"]})',
                        xaxis_title='Date',
                        yaxis_title='AQI',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No predictions found for selected model.")
            else:
                st.warning("No predictions available in MongoDB. Run inference pipeline.")
            
            st.markdown("---")
            # Historical Data (always show 90 days)
            st.markdown("## Historical Data (Last 90 Days)")
            hist_df_90 = db_handler.get_training_data(days=90)
            if hist_df_90 is not None and len(hist_df_90) > 0:
                if 'timestamp' in hist_df_90.columns and 'aqi' in hist_df_90.columns:
                    if not pd.api.types.is_datetime64_any_dtype(hist_df_90['timestamp']):
                        hist_df_90['timestamp'] = pd.to_datetime(hist_df_90['timestamp'])
                    hist_df_90 = hist_df_90.sort_values('timestamp')
                    hist_df_90['aqi'] = pd.to_numeric(hist_df_90['aqi'], errors='coerce')
                    hist_df_90 = hist_df_90.dropna(subset=['aqi'])
                    if len(hist_df_90) > 0:
                        st.info(f"Records: {len(hist_df_90)} | Range: {hist_df_90['timestamp'].min().strftime('%Y-%m-%d')} to {hist_df_90['timestamp'].max().strftime('%Y-%m-%d')}")
                        hist_fig_90 = go.Figure()
                        hist_fig_90.add_trace(go.Scatter(
                            x=hist_df_90['timestamp'],
                            y=hist_df_90['aqi'],
                            mode='lines',
                            name='AQI',
                            line=dict(color='#1f77b4', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.2)'
                        ))
                        hist_fig_90.update_layout(
                            title='AQI - Last 90 Days',
                            xaxis_title='Date',
                            yaxis_title='AQI',
                            height=400,
                            template='plotly_white'
                        )
                        st.plotly_chart(hist_fig_90, use_container_width=True)
                        # Statistics
                        st.markdown("### Statistics (90 Days)")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Average", f"{hist_df_90['aqi'].mean():.0f}")
                        s2.metric("Max", f"{hist_df_90['aqi'].max():.0f}")
                        s3.metric("Min", f"{hist_df_90['aqi'].min():.0f}")
                        s4.metric("Std Dev", f"{hist_df_90['aqi'].std():.1f}")
            else:
                st.info("No historical data available for last 90 days.")
            
            return
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    
    # Display last update time
    mongo_predictions = get_predictions_from_mongodb()
    if mongo_predictions:
        last_update = mongo_predictions.get('saved_at', 'Unknown')
        st.markdown(f"<div style='text-align: center; color: #888; font-size: 11px;'>Last prediction update: {last_update}</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            Data Sources: AQICN API, Open-Meteo (FREE) | Cache refreshes every 60 seconds | Models retrain daily<br>
            <span style="color: #888;">Historical data: Open-Meteo Air Quality API (Real data, no API key needed)</span>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
