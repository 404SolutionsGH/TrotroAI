"""
Data Pipeline Manager for Trotro AI Service
Handles data synchronization, preprocessing, and validation
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from core.database import DatabaseManager
from core.models import DatasetStatus

logger = logging.getLogger(__name__)

class DataPipelineManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_sync_time: Dict[str, datetime] = {}
        
    def sync_datasets(self, force: bool = False) -> Dict[str, DatasetStatus]:
        """Synchronize datasets from backend database"""
        status = {}
        datasets = ['stations', 'routes', 'trips', 'fares']
        
        for dataset in datasets:
            if force or self._should_sync(dataset):
                try:
                    df = self._load_dataset(dataset)
                    self.data_cache[dataset] = df
                    self.last_sync_time[dataset] = datetime.now()
                    status[dataset] = DatasetStatus.SUCCESS
                except Exception as e:
                    logger.error(f"Error syncing {dataset}: {str(e)}")
                    status[dataset] = DatasetStatus.FAILED
            else:
                status[dataset] = DatasetStatus.CACHED
        
        return status
    
    def _should_sync(self, dataset: str, max_age_hours: int = 24) -> bool:
        """Check if dataset needs to be synced"""
        if dataset not in self.last_sync_time:
            return True
            
        age = datetime.now() - self.last_sync_time[dataset]
        return age.total_seconds() > max_age_hours * 3600
    
    def _load_dataset(self, dataset: str) -> pd.DataFrame:
        """Load dataset from database"""
        if dataset == 'stations':
            return pd.DataFrame(list(self.db_manager.get_stations()))
        elif dataset == 'routes':
            return pd.DataFrame(list(self.db_manager.get_routes()))
        elif dataset == 'trips':
            return pd.DataFrame(list(self.db_manager.get_trips()))
        elif dataset == 'fares':
            return pd.DataFrame(list(self.db_manager.get_fares()))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get cached dataset with sync check"""
        if name not in self.data_cache:
            self.sync_datasets(force=True)
        return self.data_cache.get(name)
    
    def validate_data(self, df: pd.DataFrame, schema: Dict) -> bool:
        """Validate dataset against schema"""
        for column, dtype in schema.items():
            if column not in df.columns:
                return False
            if not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
                return False
        return True
    
    def preprocess_data(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Preprocess dataset for AI models"""
        if dataset == 'stations':
            return self._preprocess_stations(df)
        elif dataset == 'routes':
            return self._preprocess_routes(df)
        return df
    
    def _preprocess_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess stations dataset"""
        df['stop_lat'] = pd.to_numeric(df['station_latitude'], errors='coerce')
        df['stop_lon'] = pd.to_numeric(df['station_longitude'], errors='coerce')
        df['coordinates'] = list(zip(df['stop_lat'], df['stop_lon']))
        return df.dropna(subset=['stop_lat', 'stop_lon'])
    
    def _preprocess_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess routes dataset"""
        df['route_length'] = df.apply(
            lambda row: self._calculate_route_length(row['start_station'], row['end_station']),
            axis=1
        )
        return df
    
    def _calculate_route_length(self, start_station: str, end_station: str) -> float:
        """Calculate route length between stations"""
        # Implementation would use geospatial calculations
        return 0.0  # Placeholder

class DatasetStatus:
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    NOT_FOUND = "not_found"
