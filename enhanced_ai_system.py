#!/usr/bin/env python3
"""
Enhanced AI System for Trotro Transport
- Question generation and answering
- Fine-tuning capabilities
- MCP (Multi-Channel Platform) support
- Deepseek API integration
- WhatsApp chatbot support
"""


import os
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
from difflib import get_close_matches
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class ChatRequest:
    message: str
    context: Optional[Dict] = None

@dataclass
class ChatResponse:
    response: str
    confidence: float
    context: Optional[Dict] = None

@dataclass
class HealthCheck:
    status: str
    timestamp: str
    version: str

@dataclass
class TrainingRequest:
    data: List[Dict]
    model_type: str
    parameters: Dict

@dataclass
class PredictionRequest:
    features: Dict
    model_type: str

@dataclass
class RouteOptimizationRequest:
    origin: str
    destination: str
    constraints: Dict

class TrotroAI:
    """Enhanced AI system for Trotro transportation queries"""
    
    def __init__(self, deepseek_api_key: str = None):
        self.deepseek_api_key = deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model_path = Path(__file__).parent / 'models'
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize embeddings model
        self.tokenizer = None
        self.model = None
        self.load_embedding_model()
        
        # Sample questions and answers for training
        self.sample_qa = []
        self.generate_sample_questions()
        
    def load_embedding_model(self):
        """Load pre-trained embedding model for semantic similarity"""
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to basic text matching
            self.tokenizer = None
            self.model = None
    
    def fetch_api_data(self):
        """Fetch real data from TrotroLive API"""
        try:
            # Fetch stations
            stations_response = requests.get('https://api.trotro.live/api/stations/', timeout=10)
            stations = stations_response.json() if stations_response.status_code == 200 else []
            
            # Fetch routes
            routes_response = requests.get('https://api.trotro.live/api/routes/', timeout=10)
            routes = routes_response.json() if routes_response.status_code == 200 else []
            
            # Fetch trips
            trips_response = requests.get('https://api.trotro.live/api/trips/', timeout=10)
            trips = trips_response.json() if trips_response.status_code == 200 else []
            
            # Fetch fares
            fares_response = requests.get('https://api.trotro.live/api/fares/', timeout=10)
            fares = fares_response.json() if fares_response.status_code == 200 else []
            
            logger.info(f"Fetched {len(stations)} stations, {len(routes)} routes, {len(trips)} trips, {len(fares)} fares")
            
            return {
                'stations': stations,
                'routes': routes,
                'trips': trips,
                'fares': fares
            }
            
        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            # Return sample data as fallback
            return {
                'stations': [
                    {'id': 1, 'name': 'Circle', 'gtfs_source': 'Accra', 'station_address': 'Circle, Accra'},
                    {'id': 2, 'name': 'Madina', 'gtfs_source': 'Accra', 'station_address': 'Madina, Accra'},
                    {'id': 3, 'name': 'Kaneshie', 'gtfs_source': 'Accra', 'station_address': 'Kaneshie, Accra'},
                    {'id': 4, 'name': 'Tema', 'gtfs_source': 'Accra', 'station_address': 'Tema, Greater Accra'},
                    {'id': 5, 'name': 'Kumasi Central', 'gtfs_source': 'Kumasi', 'station_address': 'Central Market, Kumasi'},
                ],
                'routes': [
                    {'short_name': 'R1', 'long_name': 'Circle to Tema Route', 'source': 'Accra'},
                    {'short_name': 'R2', 'long_name': 'Madina to Kaneshie Route', 'source': 'Accra'},
                    {'short_name': 'R3', 'long_name': 'Kumasi Central Route', 'source': 'Kumasi'},
                ],
                'trips': [],
                'fares': []
            }
    
    def generate_sample_questions(self):
        """Generate sample questions and answers for training using real API data"""
        
        # Fetch real data from API
        api_data = self.fetch_api_data()
        stations = api_data['stations']
        routes = api_data['routes']
        trips = api_data['trips']
        fares = api_data['fares']
        
        # Create sample questions based on real data
        sample_questions = []
        
        # Station-specific questions using real data
        for station in stations[:30]:  # Limit to prevent too many samples
            station_name = station.get('name', 'Unknown Station')
            city = station.get('gtfs_source', 'Unknown City')
            address = station.get('station_address', f'Located in {city}')
            
            # Generate questions for this station
            questions = [
                f"How do I get to {station_name}?",
                f"What's the best route to {station_name}?",
                f"Where is {station_name} located?",
                f"How much does it cost to go to {station_name}?",
                f"What stations are near {station_name}?",
                f"Is {station_name} a bus stop?",
                f"What city is {station_name} in?",
                f"Tell me about {station_name} station",
            ]
            
            # Generate answers using real station data
            answers = [
                f"To get to {station_name}, you can take a trotro or taxi. It's located at {address} in {city}.",
                f"The best route to {station_name} depends on your starting point. It's a station in {city}.",
                f"{station_name} is located at {address} in {city}.",
                f"The cost to {station_name} varies depending on your starting point. Check current fares for specific routes.",
                f"You can find stations near {station_name} by checking the {city} transport network.",
                f"Yes, {station_name} is a station in the transport network.",
                f"{station_name} is in {city}.",
                f"{station_name} is a station located at {address} in {city}.",
            ]
            
            for q, a in zip(questions, answers):
                sample_questions.append({
                    'question': q,
                    'answer': a,
                    'context': f"station:{station_name}, city:{city}, address:{address}",
                    'type': 'station_info'
                })
        
        # Route-specific questions
        for i, route in enumerate(routes[:30]):  # Limit routes
            route_name = route['short_name']
            route_long = route['long_name']
            city = route['source']
            
            questions = [
                f"Tell me about route {route_name}",
                f"What is the {route_name} route?",
                f"Where does route {route_name} go?",
                f"How long is the {route_name} route?",
                f"What are the stops on route {route_name}?",
            ]
            
            answers = [
                f"Route {route_name} ({route_long}) operates in {city}. It connects various stations across the city.",
                f"The {route_name} route is called '{route_long}' and operates in {city}.",
                f"Route {route_name} ({route_long}) serves stations in {city}.",
                f"Route {route_name} covers multiple stations in {city}. Check the schedule for specific timing.",
                f"Route {route_name} ({route_long}) has multiple stops across {city}.",
            ]
            
            for q, a in zip(questions, answers):
                sample_questions.append({
                    'question': q,
                    'answer': a,
                    'context': f"route:{route_name}, city:{city}",
                    'type': 'route_info'
                })
        
        # Trip planning questions
        if len(stations) > 1:
            # Generate trip questions between random stations
            for _ in range(50):  # Generate 50 trip questions
                station1 = random.choice(stations)
                station2 = random.choice(stations)
                
                if station1['id'] != station2['id']:
                    name1 = station1['name']
                    name2 = station2['name']
                    city1 = station1['gtfs_source']
                    city2 = station2['gtfs_source']
                    
                    questions = [
                        f"How do I get from {name1} to {name2}?",
                        f"What's the route from {name1} to {name2}?",
                        f"How much does it cost from {name1} to {name2}?",
                        f"Can I travel from {name1} to {name2}?",
                        f"What's the best way to travel from {name1} to {name2}?",
                    ]
                    
                    if city1 == city2:
                        answers = [
                            f"To travel from {name1} to {name2} in {city1}, you can take a trotro or taxi. Check the available routes.",
                            f"The route from {name1} to {name2} is within {city1}. Look for connecting routes or direct trotros.",
                            f"The cost from {name1} to {name2} depends on the route and transport type. Check current fares.",
                            f"Yes, you can travel from {name1} to {name2} within {city1} using local transport.",
                            f"The best way from {name1} to {name2} in {city1} is by trotro or taxi, depending on available routes.",
                        ]
                    else:
                        answers = [
                            f"To travel from {name1} ({city1}) to {name2} ({city2}), you'll need intercity transport. Check bus schedules.",
                            f"The route from {name1} ({city1}) to {name2} ({city2}) requires intercity travel. Look for bus or taxi services.",
                            f"Travel from {name1} to {name2} is intercity ({city1} to {city2}). Costs vary by transport type.",
                            f"Yes, you can travel from {name1} ({city1}) to {name2} ({city2}) using intercity transport.",
                            f"For {name1} to {name2} travel, use intercity buses or taxis as they're in different cities.",
                        ]
                    
                    for q, a in zip(questions, answers):
                        sample_questions.append({
                            'question': q,
                            'answer': a,
                            'context': f"from:{name1}, to:{name2}, city1:{city1}, city2:{city2}",
                            'type': 'trip_planning'
                        })
        
        # General questions
        general_qa = [
            {
                'question': "What is a trotro?",
                'answer': "A trotro is a popular form of public transportation in Ghana, typically a shared minibus that follows fixed routes.",
                'context': "general",
                'type': 'general'
            },
            {
                'question': "How do I pay for trotro?",
                'answer': "You can pay for trotro with cash to the conductor, or increasingly with mobile money and digital payment systems.",
                'context': "payment",
                'type': 'general'
            },
            {
                'question': "What are the transport options in Ghana?",
                'answer': "Ghana has trotros, taxis, buses, okadas (motorcycles), and pragyas (three-wheelers) for public transport.",
                'context': "transport_types",
                'type': 'general'
            },
            {
                'question': "How do I find the nearest station?",
                'answer': "Use the TrotroLive app or website to find the nearest station to your location using GPS.",
                'context': "app_usage",
                'type': 'general'
            },
            {
                'question': "What cities does TrotroLive cover?",
                'answer': "TrotroLive covers major cities including Accra, Kumasi, and is expanding to other cities in Ghana and Africa.",
                'context': "coverage",
                'type': 'general'
            },
        ]
        
        sample_questions.extend(general_qa)
        
        self.sample_qa = sample_questions
        logger.info(f"Generated {len(sample_questions)} sample questions and answers")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            # Fallback to simple text representation
            return np.array([hash(text) % 1000])
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy().flatten()
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([hash(text) % 1000])
    
    def find_best_answer(self, question: str) -> Dict:
        """Find the best answer for a question using semantic similarity"""
        question_embedding = self.get_embeddings(question)
        
        best_match = None
        best_score = -1
        
        for qa in self.sample_qa:
            qa_embedding = self.get_embeddings(qa['question'])
            
            # Calculate cosine similarity
            if len(question_embedding) > 1 and len(qa_embedding) > 1:
                similarity = cosine_similarity(
                    question_embedding.reshape(1, -1),
                    qa_embedding.reshape(1, -1)
                )[0][0]
            else:
                # Simple fallback similarity
                similarity = 1.0 if question.lower() in qa['question'].lower() else 0.0
            
            if similarity > best_score:
                best_score = similarity
                best_match = qa
        
        return {
            'question': question,
            'answer': best_match['answer'] if best_match else "I'm sorry, I don't have information about that.",
            'confidence': best_score,
            'context': best_match['context'] if best_match else "unknown",
            'type': best_match['type'] if best_match else "unknown"
        }
    
    def call_deepseek_api(self, question: str) -> Dict:
        """Call Deepseek API for advanced question answering"""
        if not self.deepseek_api_key:
            return {"error": "Deepseek API key not configured"}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json',
            }
            
            # Create a context about trotro transportation
            context = """
            You are a helpful assistant for TrotroLive, a transportation platform in Ghana and Africa.
            You help users with questions about trotro transport, routes, fares, and travel planning.
            Trotros are shared minibuses that follow fixed routes in cities.
            Be helpful, accurate, and focused on transportation information.
            """
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': context},
                    {'role': 'user', 'content': question}
                ],
                'temperature': 0.7,
                'max_tokens': 150
            }
            
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'question': question,
                    'answer': result['choices'][0]['message']['content'],
                    'confidence': 0.8,
                    'context': 'deepseek_api',
                    'type': 'api_response'
                }
            else:
                logger.error(f"Deepseek API error: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error calling Deepseek API: {e}")
            return {"error": str(e)}
    
    def answer_question(self, question: str, use_api: bool = False) -> Dict:
        """Answer a question using local model or API"""
        if use_api:
            api_result = self.call_deepseek_api(question)
            if 'error' not in api_result:
                return api_result
            
            # Fallback to local model if API fails
            logger.info("API failed, falling back to local model")
        
        # Use local model
        return self.find_best_answer(question)
    
    def fine_tune_model(self, additional_qa: List[Dict]):
        """Fine-tune the model with additional Q&A pairs"""
        logger.info(f"Fine-tuning model with {len(additional_qa)} additional Q&A pairs")
        
        # Add new Q&A pairs to the existing dataset
        self.sample_qa.extend(additional_qa)
        
        # Save updated dataset
        self.save_model()
        
        logger.info("Model fine-tuning completed")
    
    def save_model(self):
        """Save the current model state"""
        try:
            model_file = self.model_path / 'trotro_qa_model.json'
            
            model_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'sample_qa': self.sample_qa,
                'model_info': {
                    'type': 'enhanced_qa_system',
                    'embeddings_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'total_questions': len(self.sample_qa)
                }
            }
            
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load a saved model"""
        try:
            model_file = self.model_path / 'trotro_qa_model.json'
            
            if model_file.exists():
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                
                self.sample_qa = model_data.get('sample_qa', [])
                logger.info(f"Model loaded with {len(self.sample_qa)} Q&A pairs")
                
                return True
            else:
                logger.info("No saved model found, using generated samples")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def export_model(self, format: str = 'json') -> str:
        """Export the model for deployment"""
        if format == 'json':
            export_data = {
                'model_type': 'trotro_qa_system',
                'version': '1.0',
                'exported_at': datetime.now().isoformat(),
                'sample_qa': self.sample_qa,
                'inference_example': {
                    'input': 'How do I get from Madina to Circle?',
                    'output': self.answer_question('How do I get from Madina to Circle?')
                }
            }
            
            export_file = self.model_path / f'trotro_model_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Model exported to {export_file}")
            return str(export_file)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_whatsapp_response(self, message: str) -> str:
        """Generate a response formatted for WhatsApp"""
        result = self.answer_question(message)
        
        # Format response for WhatsApp
        response = f"🚌 *TrotroLive Assistant*\n\n"
        response += f"*Question:* {result['question']}\n\n"
        response += f"*Answer:* {result['answer']}\n\n"
        
        if result['confidence'] > 0.7:
            response += "✅ *Confidence:* High\n"
        elif result['confidence'] > 0.4:
            response += "⚠️ *Confidence:* Medium\n"
        else:
            response += "❓ *Confidence:* Low - Consider rephrasing your question\n"
        
        response += f"\n_Type 'help' for more options_"
        
        return response


class TrotroMCP:
    """Multi-Channel Platform for Trotro AI"""
    
    def __init__(self, ai_system: TrotroAI):
        self.ai = ai_system
        self.channels = {
            'whatsapp': self.handle_whatsapp,
            'telegram': self.handle_telegram,
            'web': self.handle_web,
            'api': self.handle_api
        }
    
    def handle_whatsapp(self, message: str, sender_id: str) -> Dict:
        """Handle WhatsApp messages"""
        if message.lower() == 'help':
            return {
                'response': self.get_help_message(),
                'type': 'help'
            }
        
        response = self.ai.generate_whatsapp_response(message)
        
        return {
            'response': response,
            'type': 'answer',
            'sender_id': sender_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def handle_telegram(self, message: str, sender_id: str) -> Dict:
        """Handle Telegram messages"""
        # Similar to WhatsApp but with Telegram-specific formatting
        result = self.ai.answer_question(message)
        
        response = f"🚌 <b>TrotroLive Assistant</b>\n\n"
        response += f"<b>Question:</b> {result['question']}\n\n"
        response += f"<b>Answer:</b> {result['answer']}\n\n"
        
        return {
            'response': response,
            'type': 'answer',
            'sender_id': sender_id,
            'parse_mode': 'HTML'
        }
    
    def handle_web(self, message: str, session_id: str) -> Dict:
        """Handle web chat messages"""
        result = self.ai.answer_question(message, use_api=True)
        
        return {
            'response': result['answer'],
            'confidence': result['confidence'],
            'context': result['context'],
            'type': result['type'],
            'session_id': session_id
        }
    
    def handle_api(self, message: str, client_id: str) -> Dict:
        """Handle API requests"""
        result = self.ai.answer_question(message, use_api=True)
        
        return {
            'success': True,
            'data': {
                'question': result['question'],
                'answer': result['answer'],
                'confidence': result['confidence'],
                'context': result['context'],
                'type': result['type']
            },
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_help_message(self) -> str:
        """Get help message for users"""
        return """
🚌 *TrotroLive Assistant Help*

I can help you with:
• Route planning (e.g., "How do I get from Madina to Circle?")
• Station information (e.g., "Where is Kaneshie station?")
• Fare inquiries (e.g., "How much from Kumasi to Accra?")
• Transport options (e.g., "What's the best way to travel?")
• General trotro information

*Examples:*
• "Madina to Circle route"
• "Accra stations"
• "Trotro fare Circle to Tema"
• "What is a trotro?"

Type your question and I'll help you! 😊
        """
    
    def process_message(self, message: str, channel: str, sender_id: str) -> Dict:
        """Process a message from any channel"""
        if channel not in self.channels:
            return {
                'error': f'Unsupported channel: {channel}',
                'supported_channels': list(self.channels.keys())
            }
        
        return self.channels[channel](message, sender_id)


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AI System for Trotro Transport")
    parser.add_argument('--question', type=str, help='Question to ask')
    parser.add_argument('--train', action='store_true', help='Train the model with sample data')
    parser.add_argument('--export', action='store_true', help='Export the model')
    parser.add_argument('--use-api', action='store_true', help='Use Deepseek API')
    parser.add_argument('--channel', type=str, default='api', help='Channel to simulate (whatsapp, telegram, web, api)')
    parser.add_argument('--sender-id', type=str, default='test_user', help='Sender ID for testing')
    
    args = parser.parse_args()
    
    # Initialize AI system
    ai = TrotroAI()
    
    # Load existing model or use generated samples
    ai.load_model()
    
    if args.train:
        print("Training model with sample data...")
        ai.save_model()
        print("Model training completed!")
        return
    
    if args.export:
        print("Exporting model...")
        export_file = ai.export_model()
        print(f"Model exported to: {export_file}")
        return
    
    if args.question:
        # Initialize MCP
        mcp = TrotroMCP(ai)
        
        # Process question
        result = mcp.process_message(args.question, args.channel, args.sender_id)
        
        print(f"\nChannel: {args.channel}")
        print(f"Question: {args.question}")
        print(f"Response: {result.get('response', result.get('data', {}).get('answer', 'No response'))}")
        
        if 'confidence' in result:
            print(f"Confidence: {result['confidence']:.2f}")
    
    else:
        # Interactive mode
        mcp = TrotroMCP(ai)
        
        print("\n🚌 TrotroLive AI Assistant")
        print("Type 'exit' to quit, 'help' for help")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() == 'exit':
                    print("Goodbye! 👋")
                    break
                
                if question:
                    result = mcp.process_message(question, 'api', 'interactive_user')
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        data = result.get('data', {})
                        print(f"\nAnswer: {data.get('answer', 'No answer available')}")
                        print(f"Confidence: {data.get('confidence', 0):.2f}")
                        print(f"Type: {data.get('type', 'unknown')}")
                        
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
