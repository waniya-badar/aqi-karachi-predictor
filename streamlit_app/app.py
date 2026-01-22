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
        number={'font': {'size': 48, 'color': color}},
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
            
            db_handler = MongoDBHandler()
            fetcher = AQICNFetcher()
            engineer = FeatureEngineer()
            
            raw_data = fetcher.fetch_current_data()
            if not raw_data:
                st.error("Failed to fetch current AQI data")
                return
            
            current_features = engineer.create_features(raw_data)
            current_aqi = current_features['aqi']
        
        # Current AQI
        st.markdown("## Current Air Quality")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            gauge_fig, category = create_aqi_gauge(current_aqi)
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown(f"**Status: {category}**")
            st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown("### Pollutant Levels")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("PM2.5", f"{current_features.get('pm25', 0):.1f} ug/m3")
            m1.metric("PM10", f"{current_features.get('pm10', 0):.1f} ug/m3")
            m2.metric("O3", f"{current_features.get('o3', 0):.1f} ppb")
            m2.metric("NO2", f"{current_features.get('no2', 0):.1f} ppb")
            m3.metric("Temperature", f"{current_features.get('temperature', 0):.1f} C")
            m3.metric("Humidity", f"{current_features.get('humidity', 0):.0f}%")
        
        # Health Advisory
        if current_aqi > 300:
            st.error(f"HAZARDOUS - AQI: {int(current_aqi)}. Avoid outdoor activities. Stay indoors.")
        elif current_aqi > 200:
            st.error(f"Very Unhealthy - AQI: {int(current_aqi)}. Limit outdoor activities.")
        elif current_aqi > 150:
            st.warning(f"Unhealthy - AQI: {int(current_aqi)}. Sensitive groups should limit outdoor activities.")
        elif current_aqi > 100:
            st.info(f"Moderate - AQI: {int(current_aqi)}. Acceptable for most people.")
        else:
            st.success(f"Good - AQI: {int(current_aqi)}. Air quality is satisfactory.")
        
        st.markdown("---")
        
        # 3-Day Forecast
        st.markdown("## 3-Day AQI Forecast")
        
        if model and scaler and feature_names:
            predictions = predict_future_aqi(model, scaler, current_features, feature_names, days=3)
            
            if predictions:
                pred_cols = st.columns(3)
                
                for i, pred in enumerate(predictions):
                    with pred_cols[i]:
                        category, color = get_aqi_category(pred['aqi'])
                        st.markdown(f"""
                            <div style="background-color: {color}; padding: 15px; border-radius: 8px; text-align: center;">
                                <h4 style="margin: 0; color: black;">Day {pred['day']}</h4>
                                <p style="margin: 5px 0; color: black;">{pred['date']}</p>
                                <h2 style="margin: 10px 0; color: black;">{pred['aqi']:.0f}</h2>
                                <p style="margin: 0; color: black;">{category}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Forecast chart
                st.markdown("### Forecast Trend")
                
                trend_data = pd.DataFrame(predictions)
                trend_data['date'] = pd.to_datetime(trend_data['date'])
                
                current_df = pd.DataFrame([{'date': datetime.now(), 'aqi': current_aqi, 'type': 'Current'}])
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
                    title='AQI Forecast',
                    xaxis_title='Date',
                    yaxis_title='AQI',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Historical Data
        st.markdown("## Historical Data")
        
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("3 Days", use_container_width=True):
            st.session_state.history_days = 3
        if col2.button("7 Days", use_container_width=True):
            st.session_state.history_days = 7
        if col3.button("30 Days", use_container_width=True):
            st.session_state.history_days = 30
        if col4.button("90 Days", use_container_width=True):
            st.session_state.history_days = 90
        
        if 'history_days' not in st.session_state:
            st.session_state.history_days = 7
        
        history_days = st.session_state.history_days
        
        hist_df = db_handler.get_training_data(days=history_days)
        
        if hist_df is not None and len(hist_df) > 0:
            if 'timestamp' in hist_df.columns and 'aqi' in hist_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(hist_df['timestamp']):
                    hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
                
                hist_df = hist_df.sort_values('timestamp')
                hist_df['aqi'] = pd.to_numeric(hist_df['aqi'], errors='coerce')
                hist_df = hist_df.dropna(subset=['aqi'])
                
                if len(hist_df) > 0:
                    st.info(f"Records: {len(hist_df)} | Range: {hist_df['timestamp'].min().strftime('%Y-%m-%d')} to {hist_df['timestamp'].max().strftime('%Y-%m-%d')}")
                    
                    hist_fig = go.Figure()
                    hist_fig.add_trace(go.Scatter(
                        x=hist_df['timestamp'],
                        y=hist_df['aqi'],
                        mode='lines',
                        name='AQI',
                        line=dict(color='#1f77b4', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    ))
                    
                    hist_fig.update_layout(
                        title=f'AQI - Last {history_days} Days',
                        xaxis_title='Date',
                        yaxis_title='AQI',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown("### Statistics")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Average", f"{hist_df['aqi'].mean():.0f}")
                    s2.metric("Max", f"{hist_df['aqi'].max():.0f}")
                    s3.metric("Min", f"{hist_df['aqi'].min():.0f}")
                    s4.metric("Std Dev", f"{hist_df['aqi'].std():.1f}")
        else:
            st.info("No historical data available.")
        
        db_handler.close()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 12px;">
            Data: AQICN | Updated hourly | Models retrain daily
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
