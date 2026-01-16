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
    page_icon="🌫️",
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
        return "Hazardous", "#7e0023", "☠️"


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
    """Predict AQI for next N days"""
    predictions = []
    
    # Start with current features
    features_df = pd.DataFrame([current_features])
    
    for day in range(1, days + 1):
        # Select only the features used during training
        X = features_df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        predictions.append({
            'day': day,
            'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
            'aqi': max(0, pred)  # Ensure non-negative
        })
        
        # Update features for next prediction (simplified)
        # In reality, you'd update lag features properly
        features_df['hour'] = (features_df['hour'] + 24) % 24
        features_df['day'] = (features_df['day'] + 1) % 31
    
    return predictions


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🌫️ Karachi AQI Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Air Quality Index Prediction for the Next 3 Days")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ["random_forest", "gradient_boosting", "ridge"],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Data updates hourly via automated pipeline")
    
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
                st.error("❌ Failed to fetch current AQI data from API")
                return
            
            # Create features
            current_features = engineer.create_features(raw_data)
            current_aqi = current_features['aqi']
        
        # Current AQI Section
        st.markdown("## 📊 Current Air Quality")
        
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
                🚨 **HAZARDOUS ALERT - EMERGENCY CONDITIONS** 🚨
                - AQI: {} (Hazardous)
                - **IMMEDIATE ACTION REQUIRED**
                - Everyone should avoid ALL outdoor activities
                - Stay indoors with windows closed
                - Use air purifiers if available
                - Wear N95 masks if you must go outside
                - Seek medical attention if experiencing symptoms
                """.format(int(current_aqi)))
        elif current_aqi > 200:
            st.error("""
                ☠️ **VERY UNHEALTHY ALERT** ☠️
                - AQI: {} (Very Unhealthy)
                - Health alert: Everyone may experience serious health effects
                - Children, elderly, and those with respiratory conditions should remain indoors
                - General population should greatly limit outdoor activities
                - Wear masks if outdoor exposure is unavoidable
                """.format(int(current_aqi)))
        elif current_aqi > 150:
            st.warning("""
                ⚠️ **UNHEALTHY ALERT**
                - AQI: {} (Unhealthy)
                - Everyone may begin to experience health effects
                - Sensitive groups should avoid outdoor activities
                - General population should limit prolonged outdoor exertion
                - Consider wearing masks during outdoor activities
                """.format(int(current_aqi)))
        elif current_aqi > 100:
            st.info("""
                ⚠️ **MODERATE - Sensitive Groups Advisory**
                - AQI: {} (Moderate)
                - Unusually sensitive people should consider limiting prolonged outdoor exertion
                - General population can enjoy normal outdoor activities
                - Monitor symptoms if you have respiratory conditions
                """.format(int(current_aqi)))
        else:
            st.success("""
                ✅ **GOOD AIR QUALITY**
                - AQI: {} (Good)
                - It's a great day to be outdoors!
                - No health impacts expected
                - Enjoy outdoor activities
                """.format(int(current_aqi)))
        
        st.markdown("---")
        
        # Predictions Section
        st.markdown("## 🔮 3-Day AQI Forecast")
        
        if model and scaler and feature_names:
            with st.spinner("Generating predictions..."):
                predictions = predict_future_aqi(model, scaler, current_features, feature_names, days=3)
            
            # Display predictions
            pred_cols = st.columns(3)
            
            for i, pred in enumerate(predictions):
                with pred_cols[i]:
                    category, color, emoji = get_aqi_category(pred['aqi'])
                    
                    st.markdown(f"""
                        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="margin: 0; color: black;">Day {pred['day']}</h3>
                            <p style="margin: 5px 0; color: black;">{pred['date']}</p>
                            <h1 style="margin: 10px 0; color: black;">{pred['aqi']:.0f} {emoji}</h1>
                            <p style="margin: 0; color: black; font-weight: bold;">{category}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Trend chart
            st.markdown("### 📈 AQI Trend Forecast")
            
            trend_data = pd.DataFrame(predictions)
            trend_data = pd.concat([
                pd.DataFrame([{'day': 0, 'date': datetime.now().strftime('%Y-%m-%d'), 'aqi': current_aqi}]),
                trend_data
            ])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data['date'],
                y=trend_data['aqi'],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=10),
                name='Predicted AQI'
            ))
            
            # Add AQI category bands
            fig.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.1, line_width=0)
            fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.1, line_width=0)
            fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1, line_width=0)
            fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1, line_width=0)
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="AQI",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Historical Data Section
        st.markdown("## 📜 Historical AQI Data")
        
        with st.spinner("Loading historical data..."):
            hist_df = db_handler.get_training_data(days=7)
            
            if hist_df is not None and len(hist_df) > 0:
                # Plot historical AQI
                hist_fig = px.line(
                    hist_df, 
                    x='timestamp', 
                    y='aqi',
                    title='AQI - Last 7 Days',
                    labels={'timestamp': 'Date & Time', 'aqi': 'AQI'}
                )
                hist_fig.update_traces(line_color='#1f77b4', line_width=2)
                hist_fig.update_layout(height=400)
                
                st.plotly_chart(hist_fig, use_container_width=True)
                
                # Statistics
                st.markdown("### 📊 Weekly Statistics")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Average AQI", f"{hist_df['aqi'].mean():.0f}")
                with stat_col2:
                    st.metric("Max AQI", f"{hist_df['aqi'].max():.0f}")
                with stat_col3:
                    st.metric("Min AQI", f"{hist_df['aqi'].min():.0f}")
                with stat_col4:
                    st.metric("Std Dev", f"{hist_df['aqi'].std():.1f}")
            else:
                st.info("📅 Historical data will appear here once collected")
        
        # Close DB connection
        db_handler.close()
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>Data source: AQICN | Updated hourly | Made with ❤️ using Streamlit</p>
            <p>⚠️ Predictions are estimates and should not be used as the sole basis for health decisions</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()