#!/usr/bin/env python3
"""
Ghana Transport Workshop - Quick Start
Simple demonstration of the workshop capabilities
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trotrolive_webapp.settings')
import django
django.setup()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset_explorer import GhanaTransportExplorer
from route_optimizer import GhanaRouteOptimizer
from demand_predictor import GhanaDemandPredictor
from visualization import GhanaTransportVisualizer

CITY = 'accra'  # Change this to any city folder in datasets/

def quick_demo():
    """Run a quick demonstration of the workshop features"""
    print("🚀 GHANA TRANSPORT WORKSHOP - QUICK START")
    print("="*60)
    
    # 1. Basic dataset exploration
    print("\n1. 📊 DATASET EXPLORATION")
    print("-" * 30)
    explorer = GhanaTransportExplorer(city=CITY)
    explorer.basic_statistics()
    
    # 2. Route optimization demo
    print("\n2. 🤖 ROUTE OPTIMIZATION DEMO")
    print("-" * 30)
    optimizer = GhanaRouteOptimizer(city=CITY)
    optimizer.prepare_route_features()
    print("   - Route features prepared")
    print("   - Ready for AI optimization")
    
    # 3. Demand prediction demo
    print("\n3. 📈 DEMAND PREDICTION DEMO")
    print("-" * 30)
    predictor = GhanaDemandPredictor(city=CITY)
    predictor.prepare_demand_features()
    print("   - Demand features prepared")
    print("   - Ready for demand modeling")
    
    # 4. Visualization demo
    print("\n4. 🗺️ VISUALIZATION DEMO")
    print("-" * 30)
    visualizer = GhanaTransportVisualizer(city=CITY)
    visualizer.create_network_overview_map()
    print("   - Interactive map created")
    
    # 5. Key insights
    print("\n5. 💡 KEY INSIGHTS")
    print("-" * 30)
    insights = [
        f"• {CITY.title()} dataset loaded and analyzed",
        f"• {len(explorer.data['routes']) if 'routes' in explorer.data else 0} routes, {len(explorer.data['stops']) if 'stops' in explorer.data else 0} stops",
        f"• {len(explorer.data['stop_times']) if 'stop_times' in explorer.data else 0} stop times show detailed schedules",
        "• All routes are bus services (type 3) if GTFS-compliant",
        f"• Geographic coverage spans {CITY.title()} metropolitan area"
    ]
    for insight in insights:
        print(f"   {insight}")
        
    # 6. Next steps
    print("\n6. 🎯 NEXT STEPS")
    print("-" * 30)
    next_steps = [
        "Run 'python dataset_explorer.py' for full analysis",
        "Run 'python route_optimizer.py' for AI optimization",
        "Run 'python demand_predictor.py' for demand prediction",
        "Run 'python visualization.py' for interactive charts",
        "Open generated HTML files in browser"
    ]
    for step in next_steps:
        print(f"   {step}")
    print("\n" + "="*60)
    print("✅ QUICK START COMPLETE!")
    print("="*60)
    print("\nReady to explore the Ghana transport network!")
    print("Check the README.md for detailed instructions.")

def show_dataset_preview():
    """Show a preview of the dataset structure"""
    print("\n📋 DATASET PREVIEW")
    print("="*40)
    try:
        explorer = GhanaTransportExplorer(city=CITY)
        routes = explorer.data['routes']
        stops = explorer.data['stops']
        print(f"\nRoutes Data ({len(routes)} routes):")
        print(routes.head(3).to_string(index=False))
        print(f"\nStops Data ({len(stops)} stops):")
        print(stops.head(3).to_string(index=False))
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")

if __name__ == "__main__":
    # Show dataset preview
    show_dataset_preview()
    
    # Run quick demo
    quick_demo() 