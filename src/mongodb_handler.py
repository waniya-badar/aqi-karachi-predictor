"""
MongoDB Handler - Manages all database operations
This connects to MongoDB and stores/retrieves features
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()


class MongoDBHandler:
    """Handles all MongoDB operations for feature storage"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.uri = os.getenv('MONGODB_URI')
        self.db_name = os.getenv('MONGODB_DB_NAME', 'aqi_karachi')
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            print(f"Connected to MongoDB: {self.db_name}")
            
            self.db.features.create_index([("timestamp", DESCENDING)])
            self.db.features.create_index([("date", DESCENDING)])
            
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
    
    def insert_features(self, features: Dict) -> bool:
        """
        Insert processed features into MongoDB
        
        Args:
            features: Dictionary with timestamp and feature values
        
        Returns:
            bool: True if successful
        """
        try:
            # Add metadata
            features['inserted_at'] = datetime.utcnow()
            
            # Check if data for this timestamp already exists
            existing = self.db.features.find_one({
                'timestamp': features['timestamp']
            })
            
            if existing:
                self.db.features.update_one(
                    {'timestamp': features['timestamp']},
                    {'$set': features}
                )
                print(f"Updated features for {features['timestamp']}")
            else:
                self.db.features.insert_one(features)
                print(f"Inserted features for {features['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"Error inserting features: {e}")
            return False
    
    def get_latest_features(self, limit: int = 1) -> Optional[pd.DataFrame]:
        """
        Get the most recent features
        
        Args:
            limit: Number of recent records to fetch
        
        Returns:
            DataFrame with features
        """
        try:
            cursor = self.db.features.find().sort('timestamp', DESCENDING).limit(limit)
            data = list(cursor)
            
            if not data:
                print("No features found in database")
                return None
            
            for record in data:
                record.pop('_id', None)
                record.pop('inserted_at', None)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching latest features: {e}")
            return None
    
    def get_features_by_date_range(self, start_date: datetime, 
                                   end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get features within a date range
        
        Args:
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            DataFrame with features
        """
        try:
            query = {
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            cursor = self.db.features.find(query).sort('timestamp', ASCENDING)
            data = list(cursor)
            
            if not data:
                print(f"No features found between {start_date} and {end_date}")
                return None
            
            for record in data:
                record.pop('_id', None)
                record.pop('inserted_at', None)
            
            df = pd.DataFrame(data)
            print(f"Retrieved {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error fetching features by date: {e}")
            return None
    
    def get_training_data(self, days: int = 120) -> Optional[pd.DataFrame]:
        """
        Get features for model training (last N days)
        
        Args:
            days: Number of days of historical data
        
        Returns:
            DataFrame ready for training
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return self.get_features_by_date_range(start_date, end_date)
    
    def count_records(self) -> int:
        """Count total records in features collection"""
        return self.db.features.count_documents({})
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about stored data"""
        try:
            total_records = self.count_records()
            
            if total_records == 0:
                return {
                    'total_records': 0,
                    'date_range': None
                }
            
            # Get date range
            oldest = self.db.features.find_one(sort=[('timestamp', ASCENDING)])
            newest = self.db.features.find_one(sort=[('timestamp', DESCENDING)])
            
            return {
                'total_records': total_records,
                'oldest_record': oldest['timestamp'] if oldest else None,
                'newest_record': newest['timestamp'] if newest else None,
                'date_range_days': (newest['timestamp'] - oldest['timestamp']).days if oldest and newest else 0
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")


if __name__ == "__main__":
    handler = MongoDBHandler()
    stats = handler.get_data_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Records: {stats.get('total_records', 0)}")
    print(f"Date Range: {stats.get('date_range_days', 0)} days")
    handler.close()