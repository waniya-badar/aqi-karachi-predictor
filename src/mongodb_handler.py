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
            
            # Create indexes for features collection
            self.db.features.create_index([("timestamp", DESCENDING)])
            self.db.features.create_index([("date", DESCENDING)])
            
            # Create indexes for models collection
            self.db.models.create_index([("model_name", ASCENDING)], unique=True)
            self.db.models.create_index([("is_best", DESCENDING)])
            self.db.models.create_index([("trained_at", DESCENDING)])
            
            # Create indexes for training_history collection
            self.db.training_history.create_index([("timestamp", DESCENDING)])
            
            # Create indexes for predictions collection
            self.db.predictions.create_index([("timestamp", DESCENDING)])
            self.db.predictions.create_index([("saved_at", DESCENDING)])
            
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
            
            # Check if data for this HOUR already exists (avoid duplicates from different sources)
            # Round timestamp to the hour for comparison
            ts = features['timestamp']
            hour_start = ts.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            
            existing = self.db.features.find_one({
                'timestamp': {
                    '$gte': hour_start,
                    '$lt': hour_end
                }
            })
            
            if existing:
                # Update existing record for this hour
                self.db.features.update_one(
                    {'_id': existing['_id']},
                    {'$set': features}
                )
                print(f"Updated features for hour {hour_start}")
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
    
    # ==================== MODEL STORAGE METHODS ====================
    
    def save_model(self, model_name: str, model_binary: bytes, 
                   scaler_binary: bytes, feature_names: List[str],
                   metrics: Dict, is_best: bool = False) -> bool:
        """
        Save a trained model to MongoDB
        All models are stored in 'models' collection with versioning
        
        Args:
            model_name: Name of the model (e.g., 'random_forest')
            model_binary: Pickled model bytes
            scaler_binary: Pickled scaler bytes  
            feature_names: List of feature names used
            metrics: Model performance metrics
            is_best: Whether this is the best performing model
        
        Returns:
            bool: True if successful
        """
        try:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            model_doc = {
                'model_name': model_name,
                'model_binary': model_binary,
                'scaler_binary': scaler_binary,
                'feature_names': feature_names,
                'metrics': metrics,
                'is_best': is_best,
                'trained_at': datetime.utcnow(),
                'version': version
            }
            
            # Save/Update in models collection (upsert)
            self.db.models.update_one(
                {'model_name': model_name},
                {'$set': model_doc},
                upsert=True
            )
            
            print(f"[OK] Saved model '{model_name}' to MongoDB" + (" (BEST)" if is_best else ""))
            return True
            
        except Exception as e:
            print(f"[ERROR] Error saving model '{model_name}': {e}")
            return False
    
    def save_all_models(self, models_data: List[Dict]) -> bool:
        """
        Save all trained models to MongoDB
        - Latest: Saved to 'models' collection (upsert - only latest version)
        - Archive: Saved to 'models_archive' collection (all versions with timestamp)
        
        Args:
            models_data: List of dicts with model_name, model_binary, scaler_binary, 
                        feature_names, metrics, is_best
        
        Returns:
            bool: True if all successful
        """
        try:
            # Also save a training run record
            training_run = {
                'timestamp': datetime.utcnow(),
                'models': {},
                'best_model': None
            }
            
            for model_data in models_data:
                # Save to LATEST models collection (upsert)
                success = self.save_model(
                    model_name=model_data['model_name'],
                    model_binary=model_data['model_binary'],
                    scaler_binary=model_data['scaler_binary'],
                    feature_names=model_data['feature_names'],
                    metrics=model_data['metrics'],
                    is_best=model_data.get('is_best', False)
                )
                
                if not success:
                    return False
                
                # ALSO save to models_archive (all versions with timestamp)
                archive_entry = {
                    'model_name': model_data['model_name'],
                    'model_binary': model_data['model_binary'],
                    'scaler_binary': model_data['scaler_binary'],
                    'feature_names': model_data['feature_names'],
                    'metrics': model_data['metrics'],
                    'is_best': model_data.get('is_best', False),
                    'archived_at': datetime.utcnow()
                }
                self.db.models_archive.insert_one(archive_entry)
                
                # Add to training run record (without binary data)
                training_run['models'][model_data['model_name']] = model_data['metrics']
                if model_data.get('is_best'):
                    training_run['best_model'] = model_data['model_name']
            
            # Save training run to history collection
            self.db.training_history.insert_one(training_run)
            print(f"[OK] Training run saved to history")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error saving models: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Dict]:
        """
        Retrieve a model from MongoDB
        
        Args:
            model_name: Name of model to retrieve
        
        Returns:
            Dict with model_binary, scaler_binary, feature_names, metrics
        """
        try:
            model_doc = self.db.models.find_one({'model_name': model_name})
            
            if model_doc:
                model_doc.pop('_id', None)
                return model_doc
            
            print(f"Model '{model_name}' not found in database")
            return None
            
        except Exception as e:
            print(f"Error retrieving model '{model_name}': {e}")
            return None
    
    def get_best_model(self) -> Optional[Dict]:
        """
        Get the best performing model
        
        Returns:
            Dict with model data or None
        """
        try:
            model_doc = self.db.models.find_one({'is_best': True})
            
            if model_doc:
                model_doc.pop('_id', None)
                return model_doc
            
            # Fallback: get model with highest test_r2
            models = list(self.db.models.find())
            if models:
                best = max(models, key=lambda x: x.get('metrics', {}).get('test_r2', 0))
                best.pop('_id', None)
                return best
            
            return None
            
        except Exception as e:
            print(f"Error retrieving best model: {e}")
            return None
    
    def get_all_models_metadata(self) -> List[Dict]:
        """
        Get metadata for all stored models (without binary data)
        
        Returns:
            List of model metadata dicts
        """
        try:
            models = self.db.models.find({}, {
                'model_binary': 0, 
                'scaler_binary': 0
            })
            
            result = []
            for model in models:
                model.pop('_id', None)
                result.append(model)
            
            return result
            
        except Exception as e:
            print(f"Error retrieving models metadata: {e}")
            return []
    
    def get_training_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent training history
        
        Args:
            limit: Max number of records to return
        
        Returns:
            List of training run records
        """
        try:
            history = self.db.training_history.find().sort(
                'timestamp', DESCENDING
            ).limit(limit)
            
            result = []
            for record in history:
                record.pop('_id', None)
                result.append(record)
            
            return result
            
        except Exception as e:
            print(f"Error retrieving training history: {e}")
            return []
    
    def get_model_versions(self, model_name: str, limit: int = 10) -> List[Dict]:
        """
        Get model versions (note: only latest version per model is stored)
        Uses models collection which stores single latest version per model
        
        Args:
            model_name: Name of model to retrieve
            limit: Not used (only one version per model)
        
        Returns:
            List of model records (without binary data)
        """
        try:
            # Since we only store one version per model, return it if found
            model = self.db.models.find_one(
                {'model_name': model_name},
                {'model_binary': 0, 'scaler_binary': 0}
            )
            
            if model:
                model.pop('_id', None)
                return [model]
            
            return []
            
        except Exception as e:
            print(f"Error retrieving model: {e}")
            return []
    
    def save_prediction(self, prediction_data: Dict) -> bool:
        """
        Save a prediction to predictions collection in MongoDB
        
        Args:
            prediction_data: Dictionary with prediction information
        
        Returns:
            bool: True if successful
        """
        try:
            prediction_data['saved_at'] = datetime.utcnow()
            self.db.predictions.insert_one(prediction_data)
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent predictions from MongoDB
        
        Args:
            limit: Max number of predictions to return
        
        Returns:
            List of prediction records
        """
        try:
            predictions = self.db.predictions.find().sort(
                'timestamp', DESCENDING
            ).limit(limit)
            
            result = []
            for pred in predictions:
                pred.pop('_id', None)
                result.append(pred)
            
            return result
            
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return []


if __name__ == "__main__":
    handler = MongoDBHandler()
    stats = handler.get_data_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Records: {stats.get('total_records', 0)}")
    print(f"Date Range: {stats.get('date_range_days', 0)} days")

    # Check models
    models = handler.get_all_models_metadata()
    print(f"\nStored Models: {len(models)}")
    for m in models:
        print(f"  - {m['model_name']}: R²={m.get('metrics', {}).get('test_r2', 'N/A'):.4f}" +
              (" (BEST)" if m.get('is_best') else ""))

    handler.close()
