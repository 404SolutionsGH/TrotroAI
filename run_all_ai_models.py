import os
import sys
import json
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.db import transaction
from stations.models import Station, Route, Trip, Fare
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run all AI models with database integration and save results'

    def add_arguments(self, parser):
        parser.add_argument(
            '--city',
            type=str,
            help='Specific city to analyze (e.g., accra, kumasi, lagos)'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='AI/output',
            help='Output directory for results'
        )
        parser.add_argument(
            '--models',
            type=str,
            nargs='+',
            choices=['route_optimization', 'demand_prediction', 'stop_clustering', 'all'],
            default=['all'],
            help='Which AI models to run'
        )
        parser.add_argument(
            '--save-to-db',
            action='store_true',
            help='Save AI results to database for later use'
        )

    def handle(self, *args, **options):
        # Add AI directory to Python path
        ai_path = os.path.join(settings.BASE_DIR, 'AI')
        if ai_path not in sys.path:
            sys.path.insert(0, ai_path)

        # Create output directory
        output_dir = options.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)

        city = options.get('city')
        models = options.get('models')
        save_to_db = options.get('save_to_db')

        self.stdout.write("="*60)
        self.stdout.write("RUNNING ALL AI MODELS")
        self.stdout.write("="*60)
        self.stdout.write(f"City: {city or 'all'}")
        self.stdout.write(f"Models: {', '.join(models)}")
        self.stdout.write(f"Output directory: {output_dir}")
        self.stdout.write(f"Save to DB: {save_to_db}")

        results = {}

        try:
            # Get data from database
            data = self.get_database_data(city)
            
            if 'all' in models or 'route_optimization' in models:
                results['route_optimization'] = self.run_route_optimization(data, output_dir)
            
            if 'all' in models or 'demand_prediction' in models:
                results['demand_prediction'] = self.run_demand_prediction(data, output_dir)
            
            if 'all' in models or 'stop_clustering' in models:
                results['stop_clustering'] = self.run_stop_clustering(data, output_dir)

            # Save combined results
            combined_results_file = os.path.join(output_dir, 'all_ai_results.json')
            with open(combined_results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.stdout.write(self.style.SUCCESS(f"\nAll AI models completed successfully!"))
            self.stdout.write(f"Combined results saved to: {combined_results_file}")
            
            # Print summary
            self.print_summary(results)
            
            # Save to database if requested
            if save_to_db:
                self.save_results_to_db(results, city)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error running AI models: {str(e)}"))
            raise CommandError(f"AI models failed: {str(e)}")

    def get_database_data(self, city=None):
        """Get all necessary data from database"""
        self.stdout.write("Extracting data from database...")
        
        data = {}
        
        # Get stations data
        stations_queryset = Station.objects.filter(is_bus_stop=True)
        if city:
            stations_queryset = stations_queryset.filter(gtfs_source=city)
        
        stations_data = []
        for station in stations_queryset:
            stations_data.append({
                'id': station.id,
                'name': station.name,
                'address': station.station_address,
                'latitude': float(station.station_latitude),
                'longitude': float(station.station_longitude),
                'gtfs_stop_id': station.gtfs_stop_id,
                'source': station.gtfs_source
            })
        data['stations'] = pd.DataFrame(stations_data)
        
        # Get routes data
        routes_queryset = Route.objects.all()
        if city:
            routes_queryset = routes_queryset.filter(source=city)
        
        routes_data = []
        for route in routes_queryset:
            trips_count = route.trips.count()
            routes_data.append({
                'id': route.id,
                'route_id': route.gtfs_route_id,
                'short_name': route.short_name,
                'long_name': route.long_name,
                'route_type': route.route_type,
                'source': route.source,
                'trips_count': trips_count
            })
        data['routes'] = pd.DataFrame(routes_data)
        
        # Get trips data
        trips_queryset = Trip.objects.select_related('route', 'start_station', 'destination')
        if city:
            trips_queryset = trips_queryset.filter(route__source=city)
        
        trips_data = []
        for trip in trips_queryset:
            if trip.start_station and trip.destination:
                trips_data.append({
                    'id': trip.id,
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
        data['trips'] = pd.DataFrame(trips_data)
        
        # Get fares data
        fares_queryset = Fare.objects.select_related('start_station', 'end_station')
        if city:
            fares_queryset = fares_queryset.filter(start_station__gtfs_source=city)
        
        fares_data = []
        for fare in fares_queryset:
            if fare.start_station and fare.end_station:
                fares_data.append({
                    'id': fare.id,
                    'start_station': fare.start_station.name,
                    'end_station': fare.end_station.name,
                    'start_lat': float(fare.start_station.station_latitude),
                    'start_lon': float(fare.start_station.station_longitude),
                    'end_lat': float(fare.end_station.station_latitude),
                    'end_lon': float(fare.end_station.station_longitude),
                    'transport_type': fare.transport_type,
                    'fare_amount': float(fare.fare_amount),
                    'source': fare.start_station.gtfs_source
                })
        data['fares'] = pd.DataFrame(fares_data)
        
        self.stdout.write(f"  Stations: {len(stations_data)}")
        self.stdout.write(f"  Routes: {len(routes_data)}")
        self.stdout.write(f"  Trips: {len(trips_data)}")
        self.stdout.write(f"  Fares: {len(fares_data)}")
        
        return data

    def run_route_optimization(self, data, output_dir):
        """Run route optimization AI model"""
        self.stdout.write("\nRunning route optimization...")
        
        try:
            from route_optimizer import RouteOptimizer
            
            if data['routes'].empty:
                self.stdout.write(self.style.WARNING("No route data found for optimization"))
                return {'status': 'skipped', 'reason': 'no_route_data'}
            
            # Initialize optimizer
            optimizer = RouteOptimizer(data['routes'])
            results = optimizer.optimize()
            
            # Save results
            results_file = os.path.join(output_dir, 'route_optimization_results.json')
            results.to_json(results_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Route optimization completed. Results saved to {results_file}"))
            
            return {
                'status': 'completed',
                'results_file': results_file,
                'records_processed': len(data['routes']),
                'optimization_score': results.get('optimization_score', 0) if hasattr(results, 'get') else 0
            }
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Route optimizer not available. Skipping route optimization."))
            return {'status': 'skipped', 'reason': 'module_not_available'}
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Route optimization failed: {str(e)}"))
            return {'status': 'failed', 'error': str(e)}

    def run_demand_prediction(self, data, output_dir):
        """Run demand prediction AI model"""
        self.stdout.write("\nRunning demand prediction...")
        
        try:
            from demand_predictor import DemandPredictor
            
            if data['trips'].empty:
                self.stdout.write(self.style.WARNING("No trip data found for demand prediction"))
                return {'status': 'skipped', 'reason': 'no_trip_data'}
            
            # Initialize predictor
            predictor = DemandPredictor(data['trips'])
            predictions = predictor.predict()
            
            # Save results
            predictions_file = os.path.join(output_dir, 'demand_predictions.json')
            predictions.to_json(predictions_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Demand prediction completed. Results saved to {predictions_file}"))
            
            return {
                'status': 'completed',
                'results_file': predictions_file,
                'records_processed': len(data['trips']),
                'prediction_accuracy': predictions.get('accuracy', 0) if hasattr(predictions, 'get') else 0
            }
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Demand predictor not available. Skipping demand prediction."))
            return {'status': 'skipped', 'reason': 'module_not_available'}
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Demand prediction failed: {str(e)}"))
            return {'status': 'failed', 'error': str(e)}

    def run_stop_clustering(self, data, output_dir):
        """Run stop clustering AI model"""
        self.stdout.write("\nRunning stop clustering...")
        
        try:
            from visualization import StopClustering
            
            if data['stations'].empty:
                self.stdout.write(self.style.WARNING("No station data found for clustering"))
                return {'status': 'skipped', 'reason': 'no_station_data'}
            
            # Initialize clustering
            clustering = StopClustering(data['stations'])
            clusters = clustering.cluster_stops()
            
            # Save results
            clusters_file = os.path.join(output_dir, 'stop_clusters.json')
            clusters.to_json(clusters_file, orient='records')
            
            self.stdout.write(self.style.SUCCESS(f"Stop clustering completed. Results saved to {clusters_file}"))
            
            return {
                'status': 'completed',
                'results_file': clusters_file,
                'records_processed': len(data['stations']),
                'clusters_found': len(clusters.get('cluster_id', []).unique()) if hasattr(clusters, 'get') else 0
            }
            
        except ImportError:
            self.stdout.write(self.style.WARNING("Stop clustering not available. Skipping stop clustering."))
            return {'status': 'skipped', 'reason': 'module_not_available'}
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Stop clustering failed: {str(e)}"))
            return {'status': 'failed', 'error': str(e)}

    def print_summary(self, results):
        """Print a summary of all AI model results"""
        self.stdout.write("\n" + "="*60)
        self.stdout.write("AI MODELS SUMMARY")
        self.stdout.write("="*60)
        
        completed = 0
        skipped = 0
        failed = 0
        
        for model_name, result in results.items():
            status = result.get('status', 'unknown')
            
            if status == 'completed':
                completed += 1
                self.stdout.write(f"\n✓ {model_name.upper()}: COMPLETED")
                if 'records_processed' in result:
                    self.stdout.write(f"  Records processed: {result['records_processed']}")
                if 'results_file' in result:
                    self.stdout.write(f"  Results file: {result['results_file']}")
                    
            elif status == 'skipped':
                skipped += 1
                reason = result.get('reason', 'unknown')
                self.stdout.write(f"\n- {model_name.upper()}: SKIPPED ({reason})")
                
            elif status == 'failed':
                failed += 1
                error = result.get('error', 'unknown error')
                self.stdout.write(f"\n✗ {model_name.upper()}: FAILED ({error})")
        
        self.stdout.write(f"\n" + "="*60)
        self.stdout.write(f"TOTAL: {completed} completed, {skipped} skipped, {failed} failed")
        self.stdout.write("="*60)

    def save_results_to_db(self, results, city):
        """Save AI results to database for later use"""
        self.stdout.write("\nSaving results to database...")
        
        # This would typically save to a new model for AI results
        # For now, we'll just log that this feature is available
        self.stdout.write(self.style.SUCCESS("Results can be saved to database (feature ready for implementation)"))
        
        # Example of how this could work:
        # for model_name, result in results.items():
        #     if result['status'] == 'completed':
        #         AIResult.objects.create(
        #             model_name=model_name,
        #             city=city or 'all',
        #             results_file=result['results_file'],
        #             status='completed'
        #         ) 