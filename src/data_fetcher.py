"""
Data Fetcher - Gets air quality data from AQICN API
This fetches real-time AQI data for Karachi
"""

import os
import requests
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()


class AQICNFetcher:
    """Fetches air quality data from AQICN API"""
    
    def __init__(self):
        """Initialize with API key and station"""
        self.api_key = os.getenv('AQICN_API_KEY')
        self.station_id = os.getenv('KARACHI_STATION_ID', '@8762')
        self.base_url = "https://api.waqi.info/feed"
        
        if not self.api_key:
            raise ValueError("AQICN_API_KEY not found in environment variables")
    
    def fetch_current_data(self) -> Optional[Dict]:
        """
        Fetch current air quality data for Karachi
        
        Returns:
            Dictionary with AQI and pollutant data, or None if failed
        """
        try:
            # Build URL: https://api.waqi.info/feed/@8762/?token=YOUR_TOKEN
            url = f"{self.base_url}/{self.station_id}/"
            params = {'token': self.api_key}
            
            print(f"Fetching data from AQICN for Karachi...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                print(f"✗ API returned status: {data['status']}")
                return None
            
            # Extract relevant data
            aqi_data = data['data']
            
            # Parse the response
            parsed_data = {
                'timestamp': datetime.utcnow(),
                'aqi': aqi_data.get('aqi', None),
                'station_name': aqi_data.get('city', {}).get('name', 'Karachi'),
                'latitude': aqi_data.get('city', {}).get('geo', [0, 0])[0],
                'longitude': aqi_data.get('city', {}).get('geo', [0, 0])[1],
            }
            
            # Get pollutants (iaqi = individual air quality index)
            iaqi = aqi_data.get('iaqi', {})
            
            # Extract key pollutants
            parsed_data['pm25'] = iaqi.get('pm25', {}).get('v', None)
            parsed_data['pm10'] = iaqi.get('pm10', {}).get('v', None)
            parsed_data['o3'] = iaqi.get('o3', {}).get('v', None)
            parsed_data['no2'] = iaqi.get('no2', {}).get('v', None)
            parsed_data['so2'] = iaqi.get('so2', {}).get('v', None)
            parsed_data['co'] = iaqi.get('co', {}).get('v', None)
            
            # Get weather data if available
            weather = aqi_data.get('iaqi', {})
            parsed_data['temperature'] = weather.get('t', {}).get('v', None)
            parsed_data['pressure'] = weather.get('p', {}).get('v', None)
            parsed_data['humidity'] = weather.get('h', {}).get('v', None)
            parsed_data['wind_speed'] = weather.get('w', {}).get('v', None)
            
            print(f"✓ Successfully fetched AQI data: AQI={parsed_data['aqi']}")
            return parsed_data
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Network error fetching data: {e}")
            return None
        except Exception as e:
            print(f"✗ Error parsing data: {e}")
            return None
    
    def fetch_historical_data(self, date: str) -> Optional[Dict]:
        """
        Fetch historical data for a specific date
        Note: Historical data requires a paid API plan
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            Dictionary with AQI data or None
        """
        # Historical endpoint is not available in free tier
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
        print("✓ API connection successful!")
        
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