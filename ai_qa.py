#!/usr/bin/env python3
"""
AI Q&A CLI Tool for Ghana Transport
Answer questions like 'Madina to Krofrom' or 'Kumasi to Lagos' using your transport data.
Now with enhanced AI capabilities including fine-tuning and better question generation.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trotrolive_webapp.settings')
import django
django.setup()
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from stations.models import Station, Route, Trip, Fare
from difflib import get_close_matches
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(city=None):
    # Load all relevant data from ORM, filter by city if provided
    if city:
        stations = pd.DataFrame(list(Station.objects.filter(gtfs_source=city).values()))
        routes = pd.DataFrame(list(Route.objects.filter(source=city).values()))
        trips = pd.DataFrame(list(Trip.objects.filter(route__source=city).values()))
        fares = pd.DataFrame(list(Fare.objects.filter(start_station__gtfs_source=city, end_station__gtfs_source=city).values()))
    else:
        stations = pd.DataFrame(list(Station.objects.all().values()))
        routes = pd.DataFrame(list(Route.objects.all().values()))
        trips = pd.DataFrame(list(Trip.objects.all().values()))
        fares = pd.DataFrame(list(Fare.objects.all().values()))
    # Map fields for easier matching
    if not stations.empty:
        stations['stop_name'] = stations['name']
        stations['stop_id'] = stations['gtfs_stop_id'].fillna(stations['id'])
    else:
        print("[DEBUG] Stations DataFrame is empty. Columns:", stations.columns.tolist())
    if not routes.empty:
        routes['route_short_name'] = routes['short_name']
        routes['route_long_name'] = routes['long_name']
    return stations, routes, trips, fares

def find_station_by_name(name, stations):
    # Fuzzy match station name
    if 'stop_name' not in stations.columns:
        print("[DEBUG] 'stop_name' column missing in stations DataFrame. Columns:", stations.columns.tolist())
        return None
    all_names = stations['stop_name'].tolist()
    matches = get_close_matches(name, all_names, n=1, cutoff=0.6)
    if matches:
        return stations[stations['stop_name'] == matches[0]].iloc[0]
    return None

def find_route_between(station_from, station_to, routes, trips, stations):
    # Try to find a trip or route connecting the two stations
    # For now, just check if both stations are on any trip in the same city
    if trips.empty:
        return None, None
    # Find trips that start or end at the given stations
    from_id = station_from['id']
    to_id = station_to['id']
    trips_from = trips[trips['start_station_id'] == from_id]
    trips_to = trips[trips['destination_id'] == to_id]
    # Find intersection
    possible_trips = pd.merge(trips_from, trips_to, on='route_id', suffixes=('_from', '_to'))
    if not possible_trips.empty:
        # Pick the first matching route
        route_id = possible_trips.iloc[0]['route_id']
        route = routes[routes['id'] == route_id].iloc[0] if not routes.empty else None
        return route, possible_trips.iloc[0]
    return None, None

def find_fare_between(station_from, station_to, fares):
    if fares.empty:
        return None
    from_id = station_from['id']
    to_id = station_to['id']
    fare_row = fares[(fares['start_station_id'] == from_id) & (fares['end_station_id'] == to_id)]
    if not fare_row.empty:
        return fare_row.iloc[0]['fare_amount']
    return None

def main():
    parser = argparse.ArgumentParser(description="AI Q&A CLI for Ghana Transport")
    parser.add_argument('--question', type=str, required=True, help='Question, e.g. "Madina to Krofrom"')
    parser.add_argument('--city', type=str, default=None, help='City to analyze (optional)')
    args = parser.parse_args()
    print(f"Q: {args.question}")
    if args.city:
        print(f"City: {args.city}")
    stations, routes, trips, fares = load_data(args.city)
    # Parse question for origin and destination
    if ' to ' in args.question.lower():
        parts = args.question.lower().split(' to ')
        origin = parts[0].strip().title()
        destination = parts[1].strip().title()
    else:
        print("Please ask questions in the form 'Origin to Destination'.")
        return
    # Fuzzy match stations
    station_from = find_station_by_name(origin, stations)
    station_to = find_station_by_name(destination, stations)
    if station_from is None:
        print(f"Could not find station matching '{origin}'.")
        return
    if station_to is None:
        print(f"Could not find station matching '{destination}'.")
        return
    print(f"From: {station_from['stop_name']} (ID: {station_from['stop_id']})")
    print(f"To:   {station_to['stop_name']} (ID: {station_to['stop_id']})")
    # Try to find a route/trip
    route, trip = find_route_between(station_from, station_to, routes, trips, stations)
    if route is not None:
        print(f"Route found: {route['route_short_name']} - {route['route_long_name']}")
    else:
        print("No direct route found between these stations in the data.")
    # Try to find fare
    fare = find_fare_between(station_from, station_to, fares)
    if fare is not None:
        print(f"Estimated fare: {fare}")
    else:
        print("Fare not available.")

if __name__ == "__main__":
    main() 