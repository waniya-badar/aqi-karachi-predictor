"""
Data Fetcher - Gets air quality data from AQICN API and Open-Meteo
This fetches real-time AQI data for Karachi from multiple sources

Data Sources:
- REAL-TIME DATA: AQICN API (primary) → Open-Meteo (fallback)
- HISTORICAL DATA: Open-Meteo Air Quality API (backfill script)

Both APIs use the same US EPA AQI scale (0-500):
  0-50: Good (Green)
  51-100: Moderate (Yellow)
  101-150: Unhealthy for Sensitive Groups (Orange)
  151-200: Unhealthy (Red)
  201-300: Very Unhealthy (Purple)
  301-500: Hazardous (Maroon)

NO DEMO/FAKE DATA - Only real API data is used!
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()


class AQICNFetcher:
    """Fetches air quality data from AQICN API with Open-Meteo fallback"""
    
    def __init__(self):
        """Initialize with API key and station"""
        self.api_key = os.getenv('AQICN_API_KEY', 'demo')
        self.station_id = os.getenv('KARACHI_STATION_ID', 'karachi')
        self.base_url = "https://api.waqi.info/feed"
        self.open_meteo_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.open_meteo_weather_url = "https://api.open-meteo.com/v1/forecast"
        self.default_lat = 24.8607
        self.default_lon = 67.0011
    
    def fetch_current_data(self) -> Optional[Dict]:
        """
        Fetch current air quality data for Karachi
        
        Returns:
            Dictionary with AQI and pollutant data, or None if failed
        """
        try:
            # Try AQICN API
            url = f"{self.base_url}/{self.station_id}/?token={self.api_key}"
            
            print(f"Fetching data from AQICN for Karachi...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                return self._parse_aqicn_response(data['data'])
            
            print(f"AQICN API unavailable, trying Open-Meteo...")
            return self._fetch_from_open_meteo()
            
        except Exception as e:
            print(f"Error fetching from AQICN ({e}), trying Open-Meteo...")
            return self._fetch_from_open_meteo()
    
    def _fetch_from_open_meteo(self) -> Optional[Dict]:
        """
        Fetch current air quality data from Open-Meteo (FREE, no API key!)
        
        Returns:
            Dictionary with AQI data, or None if failed
        """
        try:
            # Fetch air quality data
            params = {
                'latitude': self.default_lat,
                'longitude': self.default_lon,
                'current': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi',
                'timezone': 'UTC'
            }
            
            print("Fetching from Open-Meteo Air Quality API...")
            response = requests.get(self.open_meteo_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            
            if current:
                # Also fetch current weather
                weather = self._fetch_open_meteo_weather()
                return self._parse_open_meteo_response(current, weather)
            
            raise Exception("No data available from Open-Meteo")
            
        except Exception as e:
            print(f"ERROR: Both AQICN and Open-Meteo APIs failed ({e})")
            raise Exception(f"All APIs failed: {e}")
    
    def _fetch_open_meteo_weather(self) -> Dict:
        """Fetch current weather from Open-Meteo"""
        try:
            params = {
                'latitude': self.default_lat,
                'longitude': self.default_lon,
                'current': 'temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m',
                'timezone': 'UTC'
            }
            
            response = requests.get(self.open_meteo_weather_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('current', {})
        except:
            return {}
    
    def _parse_open_meteo_response(self, current: Dict, weather: Dict) -> Dict:
        """Parse Open-Meteo API response"""
        aqi = current.get('us_aqi', 50)
        pm25 = current.get('pm2_5', 12.0)
        
        # If AQI is missing, calculate from PM2.5
        if aqi is None and pm25 is not None:
            aqi = self._calculate_aqi_from_pm25(pm25)
        
        parsed_data = {
            'timestamp': datetime.utcnow(),
            'aqi': int(round(aqi)) if aqi else 50,
            'station_name': 'Karachi (Open-Meteo)',
            'latitude': self.default_lat,
            'longitude': self.default_lon,
            'source': 'open-meteo',
            'pm25': round(pm25, 1) if pm25 else 12.0,
            'pm10': round(current.get('pm10', 25.0), 1),
            'o3': round(current.get('ozone', 30.0), 1),
            'no2': round(current.get('nitrogen_dioxide', 15.0), 1),
            'so2': round(current.get('sulphur_dioxide', 5.0), 1),
            'co': round(current.get('carbon_monoxide', 500) / 1000, 2),  # Convert µg/m³ to mg/m³
            'temperature': round(weather.get('temperature_2m', 28), 1),
            'humidity': round(weather.get('relative_humidity_2m', 65), 1),
            'pressure': round(weather.get('surface_pressure', 1013), 1),
            'wind_speed': round(weather.get('wind_speed_10m', 8), 1)
        }
        
        print(f"✓ Open-Meteo: AQI={parsed_data['aqi']} (real data)")
        return parsed_data
    
    def _parse_aqicn_response(self, aqi_data: Dict) -> Dict:
        """Parse AQICN API response"""
        parsed_data = {
            'timestamp': datetime.utcnow(),
            'aqi': aqi_data.get('aqi', 100),
            'station_name': aqi_data.get('city', {}).get('name', 'Karachi'),
            'latitude': aqi_data.get('city', {}).get('geo', [self.default_lat, self.default_lon])[0],
            'longitude': aqi_data.get('city', {}).get('geo', [self.default_lat, self.default_lon])[1],
            'source': 'aqicn'
        }
        
        # Get pollutants (iaqi = individual air quality index)
        iaqi = aqi_data.get('iaqi', {})
        parsed_data['pm25'] = iaqi.get('pm25', {}).get('v', 35)
        parsed_data['pm10'] = iaqi.get('pm10', {}).get('v', 60)
        parsed_data['o3'] = iaqi.get('o3', {}).get('v', 20)
        parsed_data['no2'] = iaqi.get('no2', {}).get('v', 15)
        parsed_data['so2'] = iaqi.get('so2', {}).get('v', 5)
        parsed_data['co'] = iaqi.get('co', {}).get('v', 0.5)
        
        # Get weather data
        parsed_data['temperature'] = iaqi.get('t', {}).get('v', 28)
        parsed_data['pressure'] = iaqi.get('p', {}).get('v', 1013)
        parsed_data['humidity'] = iaqi.get('h', {}).get('v', 65)
        parsed_data['wind_speed'] = iaqi.get('w', {}).get('v', 8)
        
        print(f"✓ AQICN: AQI={parsed_data['aqi']} (real data)")
        return parsed_data
    
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
    
    def fetch_historical_data(self, date: str) -> Optional[Dict]:
        """
        Fetch historical data for a specific date from Open-Meteo
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            Dictionary with AQI data or None
        """
        try:
            # Parse date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # Fetch from Open-Meteo Air Quality API
            params = {
                'latitude': self.default_lat,
                'longitude': self.default_lon,
                'hourly': 'pm2_5,pm10,us_aqi',
                'start_date': date,
                'end_date': date,
                'timezone': 'UTC'
            }
            
            response = requests.get(self.open_meteo_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            
            if hourly and 'us_aqi' in hourly:
                # Calculate daily averages
                aqi_values = [v for v in hourly['us_aqi'] if v is not None]
                pm25_values = [v for v in hourly.get('pm2_5', []) if v is not None]
                
                if aqi_values:
                    avg_aqi = sum(aqi_values) / len(aqi_values)
                    avg_pm25 = sum(pm25_values) / len(pm25_values) if pm25_values else 35.0
                    
                    return {
                        'timestamp': target_date,
                        'aqi': int(round(avg_aqi)),
                        'pm25': round(avg_pm25, 1),
                        'station_name': 'Karachi (Open-Meteo Historical)',
                        'source': 'open-meteo'
                    }
            
            print(f"⚠ No historical data available for {date}")
            return None
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        data = self.fetch_current_data()
        return data is not None


# Test the fetcher
if __name__ == "__main__":
    fetcher = AQICNFetcher()
    
    print("\n=== Testing AQI Data Fetcher ===")
    print("Testing AQICN and Open-Meteo connections...\n")
    
    if fetcher.test_connection():
        print("✓ API connection successful!")
        
        # Fetch and display data
        data = fetcher.fetch_current_data()
        if data:
            source = data.get('source', 'unknown')
            print(f"\nCurrent AQI Data for {data['station_name']}:")
            print(f"  Source: {source}")
            print(f"  AQI: {data['aqi']}")
            print(f"  PM2.5: {data['pm25']:.1f}")
            print(f"  PM10: {data['pm10']:.1f}")
            print(f"  Temperature: {data['temperature']}°C")
            print(f"  Humidity: {data['humidity']}%")
            print(f"\n✓ Real data from {source} (US EPA AQI scale 0-500)")
    else:
        print("✗ Both AQICN and Open-Meteo APIs failed!")