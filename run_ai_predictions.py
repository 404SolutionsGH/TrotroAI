import os
import sys
import django
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import pandas as pd
import numpy as np
from stations.models import Station, Route, Trip, Fare
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run AI predictions for transport optimization'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model',
            type=str,
            choices=['route_optimization', 'demand_prediction', 'stop_clustering', 'all'],
            default='all',
            help='Which AI model to run'
        )
        parser.add_argument(
            '--city',
            type=str,
            help='Specific city to analyze (e.g., accra, kumasi)'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='AI/output',
            help='Output directory for results'
        )

    def handle(self, *args, **options):
        # Add AI directory to Python path
        ai_path = os.path.join(settings.BASE_DIR, 'AI')
        if ai_path not in sys.path:
            sys.path.insert(0, ai_path)

        # Create output directory
        output_dir = options.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)

        model = options.get('model')
        city = options.get('city')

        self.stdout.write(f"Running AI predictions: {model}")
        self.stdout.write(f"Output directory: {output_dir}")

        try:
            if model in ['route_optimization', 'all']:
                self.run_route_optimization(city, output_dir)
            
            if model in ['demand_prediction', 'all']:
                self.run_demand_prediction(city, output_dir)
            
            if model in ['stop_clustering', 'all']:
                self.run_stop_clustering(city, output_dir)

            self.stdout.write(self.style.SUCCESS("AI predictions completed successfully!"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error running AI predictions: {str(e)}"))
            raise CommandError(f"AI prediction failed: {str(e)}")

    def run_route_optimization(self, city, output_dir):
        """Run route optimization AI model"""
        self.stdout.write("Running route optimization...")
        
        try:
            # Import the route optimizer
            from route_optimizer import RouteOptimizer
            
            # Get data from database
            routes_data = self.get_routes_data(city)
            
            if routes_data.empty:
                self.stdout.write(self.style.WARNING("No route data found for optimization"))
                return
            
            # Initialize and run optimizer
            optimizer = RouteOptimizer(routes_data)
            results = optimizer.optimize()
            
            # Save results
            results_file = os.path.join(output_dir, 'route_optimization_results.json')
            results.to_json(results_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Route optimization completed. Results saved to {results_file}"))
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Route optimizer not available. Skipping route optimization."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Route optimization failed: {str(e)}"))

    def run_demand_prediction(self, city, output_dir):
        """Run demand prediction AI model"""
        self.stdout.write("Running demand prediction...")
        
        try:
            # Import the demand predictor
            from demand_predictor import DemandPredictor
            
            # Get data from database
            trips_data = self.get_trips_data(city)
            
            if trips_data.empty:
                self.stdout.write(self.style.WARNING("No trip data found for demand prediction"))
                return
            
            # Initialize and run predictor
            predictor = DemandPredictor(trips_data)
            predictions = predictor.predict()
            
            # Save results
            predictions_file = os.path.join(output_dir, 'demand_predictions.json')
            predictions.to_json(predictions_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Demand prediction completed. Results saved to {predictions_file}"))
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Demand predictor not available. Skipping demand prediction."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Demand prediction failed: {str(e)}"))

    def run_stop_clustering(self, city, output_dir):
        """Run stop clustering AI model"""
        self.stdout.write("Running stop clustering...")
        
        try:
            # Import the visualization module for clustering
            from visualization import StopClustering
            
            # Get data from database
            stops_data = self.get_stops_data(city)
            
            if stops_data.empty:
                self.stdout.write(self.style.WARNING("No stop data found for clustering"))
                return
            
            # Initialize and run clustering
            clustering = StopClustering(stops_data)
            clusters = clustering.cluster_stops()
            
            # Save results
            clusters_file = os.path.join(output_dir, 'stop_clusters.json')
            clusters.to_json(clusters_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Stop clustering completed. Results saved to {clusters_file}"))
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Stop clustering not available. Skipping stop clustering."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Stop clustering failed: {str(e)}"))

    def get_routes_data(self, city=None):
        """Get routes data from database"""
        queryset = Route.objects.all()
        
        if city:
            queryset = queryset.filter(source=city)
        
        data = []
        for route in queryset:
            # Get related trips for additional metrics
            trips_count = route.trips.count()
            
            data.append({
                'route_id': route.gtfs_route_id,
                'short_name': route.short_name,
                'long_name': route.long_name,
                'route_type': route.route_type,
                'source': route.source,
                'trips_count': trips_count
            })
        
        return pd.DataFrame(data)

    def get_trips_data(self, city=None):
        """Get trips data from database"""
        queryset = Trip.objects.select_related('route', 'start_station', 'destination')
        
        if city:
            queryset = queryset.filter(route__source=city)
        
        data = []
        for trip in queryset:
            if trip.start_station and trip.destination:
                data.append({
                    'trip_id': trip.gtfs_trip_id,
                    'route_id': trip.gtfs_route_id,
                    'start_station': trip.start_station.name,
                    'end_station': trip.destination.name,
                    'start_lat': float(trip.start_station.station_latitude),
                    'start_lon': float(trip.start_station.station_longitude),
                    'end_lat': float(trip.destination.station_latitude),
                    'end_lon': float(trip.destination.station_longitude),
                    'transport_type': trip.transport_type,
                    'source': trip.route.source if trip.route else 'unknown'
                })
        
        return pd.DataFrame(data)

    def get_stops_data(self, city=None):
        """Get stops data from database"""
        queryset = Station.objects.filter(is_bus_stop=True)
        
        if city:
            queryset = queryset.filter(gtfs_source=city)
        
        data = []
        for stop in queryset:
            data.append({
                'stop_id': stop.gtfs_stop_id,
                'name': stop.name,
                'address': stop.station_address,
                'latitude': float(stop.station_latitude),
                'longitude': float(stop.station_longitude),
                'source': stop.gtfs_source
            })
        
        return pd.DataFrame(data) 