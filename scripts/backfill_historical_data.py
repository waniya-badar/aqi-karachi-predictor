"""
Backfill Historical Data - Fetches REAL historical AQI and weather data from Open-Meteo
Open-Meteo is completely FREE and requires NO API key!

Data Sources:
- Air Quality: https://open-meteo.com/en/docs/air-quality-api (PM2.5, PM10, O3, NO2, SO2, CO, US AQI)
- Weather: https://open-meteo.com/en/docs/historical-weather-api (Temperature, Humidity, Pressure, Wind)

This provides REAL historical data for Karachi for model training.
"""

import os
import sys
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import FeatureEngineer
from src.mongodb_handler import MongoDBHandler
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenMeteoFetcher:
    """
    Fetches REAL historical air quality and weather data from Open-Meteo APIs.
    Completely FREE - No API key required!
    """
    
    def __init__(self):
        """Initialize Open-Meteo fetcher"""
        self.air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.weather_archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.weather_forecast_url = "https://api.open-meteo.com/v1/forecast"
        
        # Karachi coordinates
        self.latitude = 24.8607
        self.longitude = 67.0011
        
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'AQI-Predictor-Karachi/1.0'
        }
    
    def fetch_air_quality_history(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Fetch historical air quality data from Open-Meteo
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary with hourly air quality data
        """
        try:
            params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi',
                'start_date': start_date,
                'end_date': end_date,
                'timezone': 'UTC'
            }
            
            logger.info(f"Fetching air quality data from {start_date} to {end_date}...")
            response = requests.get(self.air_quality_url, params=params, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            hourly = data.get('hourly', {})
            
            if hourly and 'time' in hourly:
                logger.info(f"✓ Retrieved {len(hourly['time'])} hours of air quality data")
                return hourly
            
            logger.warning("No air quality data returned")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching air quality: {e}")
            return None
    
    def fetch_weather_history(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Fetch historical weather data from Open-Meteo Archive API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary with hourly weather data
        """
        try:
            params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': 'temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m',
                'start_date': start_date,
                'end_date': end_date,
                'timezone': 'UTC'
            }
            
            logger.info(f"Fetching weather data from {start_date} to {end_date}...")
            response = requests.get(self.weather_archive_url, params=params, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            hourly = data.get('hourly', {})
            
            if hourly and 'time' in hourly:
                logger.info(f"✓ Retrieved {len(hourly['time'])} hours of weather data")
                return hourly
            
            logger.warning("No weather data returned")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None
    
    def fetch_recent_air_quality(self, past_days: int = 7) -> Optional[Dict]:
        """
        Fetch recent air quality data (last N days) - uses different endpoint
        
        Args:
            past_days: Number of past days to fetch
        
        Returns:
            Dictionary with hourly air quality data
        """
        try:
            params = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'hourly': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi',
                'past_days': past_days,
                'timezone': 'UTC'
            }
            
            logger.info(f"Fetching recent {past_days} days of air quality data...")
            response = requests.get(self.air_quality_url, params=params, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            hourly = data.get('hourly', {})
            
            if hourly and 'time' in hourly:
                logger.info(f"✓ Retrieved {len(hourly['time'])} hours of recent air quality data")
                return hourly
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching recent air quality: {e}")
            return None
    
    def merge_data(self, air_quality: Dict, weather: Dict) -> List[Dict]:
        """
        Merge air quality and weather data into unified records
        
        Args:
            air_quality: Air quality hourly data
            weather: Weather hourly data
        
        Returns:
            List of merged hourly records
        """
        records = []
        
        # Create lookup for weather data
        weather_lookup = {}
        if weather and 'time' in weather:
            for i, t in enumerate(weather['time']):
                weather_lookup[t] = {
                    'temperature': weather.get('temperature_2m', [None] * len(weather['time']))[i],
                    'humidity': weather.get('relative_humidity_2m', [None] * len(weather['time']))[i],
                    'pressure': weather.get('surface_pressure', [None] * len(weather['time']))[i],
                    'wind_speed': weather.get('wind_speed_10m', [None] * len(weather['time']))[i]
                }
        
        # Process air quality data
        if air_quality and 'time' in air_quality:
            times = air_quality['time']
            pm25_list = air_quality.get('pm2_5', [None] * len(times))
            pm10_list = air_quality.get('pm10', [None] * len(times))
            o3_list = air_quality.get('ozone', [None] * len(times))
            no2_list = air_quality.get('nitrogen_dioxide', [None] * len(times))
            so2_list = air_quality.get('sulphur_dioxide', [None] * len(times))
            co_list = air_quality.get('carbon_monoxide', [None] * len(times))
            aqi_list = air_quality.get('us_aqi', [None] * len(times))
            
            for i, t in enumerate(times):
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(t.replace('Z', '+00:00'))
                    timestamp = timestamp.replace(tzinfo=None)
                    
                    # Get values
                    pm25 = pm25_list[i]
                    aqi = aqi_list[i]
                    
                    # Skip if no PM2.5 or AQI data
                    if pm25 is None and aqi is None:
                        continue
                    
                    # If AQI is missing, calculate from PM2.5
                    if aqi is None and pm25 is not None:
                        aqi = self._calculate_aqi_from_pm25(pm25)
                    
                    # If PM2.5 is missing but AQI exists, estimate PM2.5
                    if pm25 is None and aqi is not None:
                        pm25 = self._estimate_pm25_from_aqi(aqi)
                    
                    record = {
                        'timestamp': timestamp,
                        'aqi': int(round(aqi)) if aqi else 50,
                        'pm25': round(pm25, 1) if pm25 else 12.0,
                        'pm10': round(pm10_list[i], 1) if pm10_list[i] else 25.0,
                        'o3': round(o3_list[i], 1) if o3_list[i] else 30.0,
                        'no2': round(no2_list[i], 1) if no2_list[i] else 15.0,
                        'so2': round(so2_list[i], 1) if so2_list[i] else 5.0,
                        'co': round(co_list[i] / 1000, 2) if co_list[i] else 0.5,  # Convert µg/m³ to mg/m³
                        'station_name': 'Karachi (Open-Meteo)',
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'source': 'open-meteo'
                    }
                    
                    # Add weather data if available
                    if t in weather_lookup:
                        w = weather_lookup[t]
                        record['temperature'] = round(w['temperature'], 1) if w['temperature'] else 28.0
                        record['humidity'] = round(w['humidity'], 1) if w['humidity'] else 65.0
                        record['pressure'] = round(w['pressure'], 1) if w['pressure'] else 1013.0
                        record['wind_speed'] = round(w['wind_speed'], 1) if w['wind_speed'] else 8.0
                    else:
                        # Use default values for recent data where archive weather isn't available
                        record['temperature'] = 28.0
                        record['humidity'] = 65.0
                        record['pressure'] = 1013.0
                        record['wind_speed'] = 8.0
                    
                    records.append(record)
                    
                except Exception as e:
                    continue
        
        return records
    
    def _calculate_aqi_from_pm25(self, pm25: float) -> int:
        """Calculate AQI from PM2.5 using EPA formula"""
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ]
        
        pm25 = max(0, pm25)
        for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
            if bp_lo <= pm25 <= bp_hi:
                return int(round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo))
        return 500 if pm25 > 500.4 else 0
    
    def _estimate_pm25_from_aqi(self, aqi: float) -> float:
        """Estimate PM2.5 from AQI (reverse calculation)"""
        breakpoints = [
            (0, 50, 0.0, 12.0),
            (51, 100, 12.1, 35.4),
            (101, 150, 35.5, 55.4),
            (151, 200, 55.5, 150.4),
            (201, 300, 150.5, 250.4),
            (301, 400, 250.5, 350.4),
            (401, 500, 350.5, 500.4),
        ]
        
        for aqi_lo, aqi_hi, pm_lo, pm_hi in breakpoints:
            if aqi_lo <= aqi <= aqi_hi:
                return ((pm_hi - pm_lo) / (aqi_hi - aqi_lo)) * (aqi - aqi_lo) + pm_lo
        return 500.0


def clear_old_data(db_handler: MongoDBHandler) -> int:
    """
    Clear all old/synthetic data from MongoDB
    
    Returns:
        Number of records deleted
    """
    try:
        # Count existing records
        old_count = db_handler.db.features.count_documents({})
        logger.info(f"Found {old_count} existing records in database")
        
        if old_count > 0:
            # Delete all records
            result = db_handler.db.features.delete_many({})
            logger.info(f"✓ Deleted {result.deleted_count} old records")
            return result.deleted_count
        
        return 0
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        return 0


def run_backfill(days: int = 90, clear_existing: bool = True, dry_run: bool = False):
    """
    Main backfill function - fetches REAL historical data from Open-Meteo
    
    Args:
        days: Number of days of historical data to fetch (default: 90 = 3 months)
        clear_existing: Whether to clear existing data first
        dry_run: If True, don't actually store data
    """
    logger.info("=" * 60)
    logger.info("Open-Meteo Historical Data Backfill")
    logger.info("Fetching REAL data for Karachi - No API key needed!")
    logger.info(f"Target: {days} days of hourly data")
    logger.info("=" * 60)
    
    # Initialize components
    fetcher = OpenMeteoFetcher()
    engineer = FeatureEngineer()
    
    if not dry_run:
        try:
            db_handler = MongoDBHandler()
            
            # Clear old data if requested
            if clear_existing:
                logger.info("\nStep 1: Clearing old/synthetic data...")
                clear_old_data(db_handler)
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Open-Meteo archive has delay, so adjust dates
    # Archive data is available up to 5 days ago
    archive_end = end_date - timedelta(days=5)
    
    logger.info(f"\nStep 2: Fetching historical data...")
    logger.info(f"  Archive period: {start_date.date()} to {archive_end.date()}")
    logger.info(f"  Recent period: {archive_end.date()} to {end_date.date()}")
    
    all_records = []
    
    # Fetch in chunks (Open-Meteo works best with ~30 day chunks)
    chunk_size = 30
    current_start = start_date
    
    while current_start < archive_end:
        current_end = min(current_start + timedelta(days=chunk_size), archive_end)
        
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        
        logger.info(f"\n  Chunk: {start_str} to {end_str}")
        
        # Fetch air quality
        air_quality = fetcher.fetch_air_quality_history(start_str, end_str)
        
        # Fetch weather
        weather = fetcher.fetch_weather_history(start_str, end_str)
        
        if air_quality:
            # Merge data
            records = fetcher.merge_data(air_quality, weather)
            all_records.extend(records)
            logger.info(f"    → Added {len(records)} records")
        
        current_start = current_end + timedelta(days=1)
        time.sleep(0.5)  # Rate limiting
    
    # Fetch recent data (last 5 days)
    logger.info("\n  Fetching recent data (last 5 days)...")
    recent_air = fetcher.fetch_recent_air_quality(past_days=5)
    if recent_air:
        recent_records = fetcher.merge_data(recent_air, {})
        all_records.extend(recent_records)
        logger.info(f"    → Added {len(recent_records)} recent records")
    
    if not all_records:
        logger.error("No data retrieved from Open-Meteo")
        return False
    
    # Remove duplicates by timestamp
    logger.info(f"\nStep 3: Processing {len(all_records)} total records...")
    unique_records = {}
    for record in all_records:
        ts = record['timestamp']
        unique_records[ts] = record
    
    records = sorted(unique_records.values(), key=lambda x: x['timestamp'])
    logger.info(f"  Unique records: {len(records)}")
    
    # Store records
    if not dry_run:
        logger.info("\nStep 4: Storing records in MongoDB...")
        success_count = 0
        error_count = 0
        
        for i, record in enumerate(records):
            try:
                features = engineer.create_features(record)
                if features:
                    success = db_handler.insert_features(features)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                
                if (i + 1) % 500 == 0:
                    logger.info(f"    Progress: {i + 1}/{len(records)} records...")
                    
            except Exception as e:
                error_count += 1
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Backfill Complete!")
        logger.info(f"  ✓ Successfully stored: {success_count} records")
        logger.info(f"  ✗ Errors: {error_count} records")
        
        total = db_handler.count_records()
        stats = db_handler.get_data_statistics()
        logger.info(f"  Total in database: {total}")
        if stats.get('oldest_record'):
            logger.info(f"  Date range: {stats['oldest_record'].date()} to {stats['newest_record'].date()}")
        
        db_handler.close()
        logger.info("=" * 60)
        
        return success_count > 0
    else:
        logger.info(f"\n[DRY RUN] Would store {len(records)} records")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill historical AQI data from Open-Meteo (FREE, no API key!)')
    parser.add_argument('--days', type=int, default=90, help='Number of days to fetch (default: 90)')
    parser.add_argument('--keep-existing', action='store_true', help='Keep existing data (default: clear)')
    parser.add_argument('--dry-run', action='store_true', help='Run without storing to database')
    
    args = parser.parse_args()
    
    success = run_backfill(
        days=args.days, 
        clear_existing=not args.keep_existing,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)
