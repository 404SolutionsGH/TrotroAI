from core.trotro_api import TrotroAPIService
from datetime import datetime
import os
import json
from enhanced_ai_system import TrotroAI

def fetch_and_process_data():
    # Initialize API service
    api_service = TrotroAPIService()
    
    try:
        # Fetch all data from API
        print("Fetching data from Trotro API...")
        data = api_service.fetch_all_data()
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_filename = f'data/trotro_raw_data_{timestamp}.json'
        api_service.save_data_to_file(raw_filename)
        
        # Initialize AI system
        ai_system = TrotroAI()
        
        # Process and train models with new data
        print("Processing data and training models...")
        ai_system.process_new_data(data)
        ai_system.train_models()
        
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during data fetching or training: {str(e)}")
        raise

if __name__ == '__main__':
    fetch_and_process_data()
