import requests
import json
from typing import Dict, List, Any
from datetime import datetime
import os

class TrotroAPIService:
    def __init__(self):
        self.base_url = "https://api.trotro.live"
        self.api_key = os.getenv('TROTRO_API_KEY')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}' if self.api_key else '',
            'Content-Type': 'application/json'
        }

    def get_stations(self) -> List[Dict[str, Any]]:
        """Fetch all stations data"""
        url = f"{self.base_url}/stations"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_routes(self) -> List[Dict[str, Any]]:
        """Fetch all routes data"""
        url = f"{self.base_url}/routes"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_trips(self, date: str = None) -> List[Dict[str, Any]]:
        """Fetch trip data for a specific date or all trips"""
        url = f"{self.base_url}/trips"
        params = {}
        if date:
            params['date'] = date
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_fares(self) -> List[Dict[str, Any]]:
        """Fetch fare information"""
        url = f"{self.base_url}/fares"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_realtime_info(self) -> Dict[str, Any]:
        """Fetch real-time information"""
        url = f"{self.base_url}/realtime"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all available data from the API"""
        data = {
            'stations': self.get_stations(),
            'routes': self.get_routes(),
            'trips': self.get_trips(),
            'fares': self.get_fares(),
            'realtime': self.get_realtime_info()
        }
        return data

    def save_data_to_file(self, filename: str = 'trotro_dataset.json') -> None:
        """Save fetched data to a JSON file"""
        data = self.fetch_all_data()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Data saved to {filename}")
