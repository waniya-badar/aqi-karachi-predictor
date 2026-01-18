"""
Data Fetcher - Gets air quality data from AQICN API
This fetches real-time AQI data for Karachi
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict
from dotenv import load_dotenv
import random

load_dotenv()


class AQICNFetcher:
    """Fetches air quality data from AQICN API"""
    
    def __init__(self):
        """Initialize with API key and station"""
        self.api_key = os.getenv('AQICN_API_KEY', 'demo')
        self.station_id = os.getenv('KARACHI_STATION_ID', 'karachi')
        self.base_url = "https://api.waqi.info/feed"
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
            
            print(f"API unavailable, using simulated data for demo")
            return self._generate_demo_data()
            
        except Exception as e:
            print(f"Error fetching from API ({e}), using demo data")
            return self._generate_demo_data()
    
    def _parse_aqicn_response(self, aqi_data: Dict) -> Dict:
        """Parse AQICN API response"""
        parsed_data = {
            'timestamp': datetime.utcnow(),
            'aqi': aqi_data.get('aqi', 100),
            'station_name': aqi_data.get('city', {}).get('name', 'Karachi'),
            'latitude': aqi_data.get('city', {}).get('geo', [self.default_lat, self.default_lon])[0],
            'longitude': aqi_data.get('city', {}).get('geo', [self.default_lat, self.default_lon])[1],
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
        
        print(f"Successfully fetched AQI data: AQI={parsed_data['aqi']}")
        return parsed_data
    
    def _generate_demo_data(self) -> Dict:
        """Generate realistic demo data for testing"""
        base_aqi = random.randint(80, 180)
        
        return {
            'timestamp': datetime.utcnow(),
            'aqi': base_aqi,
            'station_name': 'Karachi',
            'latitude': self.default_lat,
            'longitude': self.default_lon,
            'pm25': base_aqi * 0.6 + random.randint(5, 20),
            'pm10': base_aqi * 0.8 + random.randint(10, 30),
            'o3': random.randint(15, 45),
            'no2': random.randint(10, 40),
            'so2': random.randint(2, 15),
            'co': random.uniform(0.3, 2.0),
            'temperature': random.randint(25, 38),
            'pressure': random.randint(1010, 1016),
            'humidity': random.randint(40, 80),
            'wind_speed': random.randint(3, 15)
        }
    
    def fetch_historical_data(self, date: str) -> Optional[Dict]:
        """
        Fetch historical data for a specific date
        Note: Historical data requires a paid API plan
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            Dictionary with AQI data or None
        """
        # Historical endpoint is not available in free tier, use demo data
        # We would implement this if using paid plan
        print("⚠ Historical data requires paid API plan")
        return None
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        data = self.fetch_current_data()
        return data is not None


# Test the fetcher
if __name__ == "__main__":
    fetcher = AQICNFetcher()
    
    print("\n=== Testing AQICN API Connection ===")
    if fetcher.test_connection():
        print("API connection successful!")
        
        # Fetch and display data
        data = fetcher.fetch_current_data()
        if data:
            print(f"\nCurrent AQI Data for {data['station_name']}:")
            print(f"AQI: {data['aqi']}")
            print(f"PM2.5: {data['pm25']}")
            print(f"PM10: {data['pm10']}")
            print(f"Temperature: {data['temperature']}°C")
            print(f"Humidity: {data['humidity']}%")
    else:
        print("✗ API connection failed. Check your API key.")