#!/usr/bin/env python3
"""
WhatsApp Integration for TrotroLive AI
- Webhook for WhatsApp Business API
- Integration with enhanced AI system
- Message handling and response formatting
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trotrolive_webapp.settings')
import django
django.setup()

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any
from .enhanced_ai_system import TrotroAI, TrotroMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppService:
    """WhatsApp Business API integration service"""
    
    def __init__(self, access_token: str = None, phone_number_id: str = None):
        self.access_token = access_token or os.getenv('WHATSAPP_ACCESS_TOKEN')
        self.phone_number_id = phone_number_id or os.getenv('WHATSAPP_PHONE_NUMBER_ID')
        self.api_url = f"https://graph.facebook.com/v17.0/{self.phone_number_id}/messages"
        
        # Initialize AI system
        self.ai = TrotroAI()
        self.ai.load_model()
        self.mcp = TrotroMCP(self.ai)
        
        # Message templates
        self.templates = {
            'welcome': """
🚌 *Welcome to TrotroLive!*

I'm your AI assistant for trotro transport in Ghana. I can help you with:

• Route planning
• Station information  
• Fare inquiries
• Transport options
• General trotro info

*Examples:*
• "How do I get from Madina to Circle?"
• "Where is Kaneshie station?"
• "What's the fare from Kumasi to Accra?"

Type your question and I'll help! 😊
            """,
            'error': """
😔 *Sorry, something went wrong!*

Please try again or contact support if the issue persists.

Type *help* for assistance.
            """
        }
    
    def verify_webhook(self, request):
        """Verify WhatsApp webhook"""
        try:
            mode = request.GET.get('hub.mode')
            token = request.GET.get('hub.verify_token')
            challenge = request.GET.get('hub.challenge')
            
            # Verify token (you should set this in your environment)
            verify_token = os.getenv('WHATSAPP_VERIFY_TOKEN', 'your_verify_token')
            
            if mode == 'subscribe' and token == verify_token:
                logger.info("WhatsApp webhook verified successfully")
                return HttpResponse(challenge)
            else:
                logger.error("WhatsApp webhook verification failed")
                return HttpResponse("Verification failed", status=403)
                
        except Exception as e:
            logger.error(f"Error verifying webhook: {e}")
            return HttpResponse("Error", status=500)
    
    def send_message(self, to: str, message: str, message_type: str = 'text') -> Dict[str, Any]:
        """Send a message via WhatsApp Business API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": message_type,
                "text": {
                    "body": message
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {to}")
                return {"success": True, "data": response.json()}
            else:
                logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}
    
    def send_template_message(self, to: str, template_name: str, language: str = 'en') -> Dict[str, Any]:
        """Send a template message"""
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {
                        "code": language
                    }
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Template message sent successfully to {to}")
                return {"success": True, "data": response.json()}
            else:
                logger.error(f"Failed to send template message: {response.status_code}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error sending template message: {e}")
            return {"success": False, "error": str(e)}
    
    def process_incoming_message(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming WhatsApp message"""
        try:
            # Extract message data
            entry = webhook_data.get('entry', [{}])[0]
            changes = entry.get('changes', [{}])[0]
            value = changes.get('value', {})
            
            # Check if it's a message
            if 'messages' not in value:
                return {"success": True, "message": "No messages to process"}
            
            messages = value['messages']
            contacts = value.get('contacts', [])
            
            for message in messages:
                # Extract message info
                message_id = message.get('id')
                sender = message.get('from')
                message_type = message.get('type')
                timestamp = message.get('timestamp')
                
                # Get sender info
                sender_name = "Unknown"
                if contacts:
                    for contact in contacts:
                        if contact.get('wa_id') == sender:
                            sender_name = contact.get('profile', {}).get('name', 'Unknown')
                            break
                
                # Process text messages
                if message_type == 'text':
                    text_body = message.get('text', {}).get('body', '')
                    
                    logger.info(f"Received message from {sender} ({sender_name}): {text_body}")
                    
                    # Handle special commands
                    if text_body.lower() in ['hi', 'hello', 'start', 'help']:
                        response = self.templates['welcome']
                    elif text_body.lower() == 'help':
                        response = self.mcp.get_help_message()
                    else:
                        # Process with AI
                        try:
                            ai_result = self.mcp.process_message(text_body, 'whatsapp', sender)
                            response = ai_result.get('response', self.templates['error'])
                        except Exception as e:
                            logger.error(f"AI processing error: {e}")
                            response = self.templates['error']
                    
                    # Send response
                    send_result = self.send_message(sender, response)
                    
                    if send_result['success']:
                        logger.info(f"Response sent to {sender}")
                    else:
                        logger.error(f"Failed to send response to {sender}")
                
                # Handle other message types
                elif message_type in ['image', 'document', 'audio', 'video']:
                    response = "📎 I can only process text messages right now. Please send your question as text."
                    self.send_message(sender, response)
                
                elif message_type == 'location':
                    response = "📍 Thanks for sharing your location! I'll help you find nearby stations and routes."
                    self.send_message(sender, response)
                
                else:
                    response = "🤔 I didn't understand that message type. Please send a text message with your question."
                    self.send_message(sender, response)
            
            return {"success": True, "processed": len(messages)}
            
        except Exception as e:
            logger.error(f"Error processing incoming message: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_webhook(self, request):
        """Handle WhatsApp webhook"""
        try:
            # GET request for verification
            if request.method == 'GET':
                return self.verify_webhook(request)
            
            # POST request for messages
            elif request.method == 'POST':
                webhook_data = json.loads(request.body)
                result = self.process_incoming_message(webhook_data)
                
                if result['success']:
                    return JsonResponse({"status": "success"})
                else:
                    return JsonResponse({"status": "error", "message": result.get('error')})
            
            else:
                return HttpResponse("Method not allowed", status=405)
                
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)


# Django views
@method_decorator(csrf_exempt, name='dispatch')
class WhatsAppWebhookView(View):
    """Django view for WhatsApp webhook"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whatsapp_service = WhatsAppService()
    
    def get(self, request):
        """Handle webhook verification"""
        return self.whatsapp_service.verify_webhook(request)
    
    def post(self, request):
        """Handle incoming messages"""
        return self.whatsapp_service.handle_webhook(request)


# Utility functions
def send_whatsapp_message(phone_number: str, message: str) -> Dict[str, Any]:
    """Utility function to send WhatsApp message"""
    service = WhatsAppService()
    return service.send_message(phone_number, message)


def broadcast_message(phone_numbers: list, message: str) -> Dict[str, Any]:
    """Broadcast message to multiple users"""
    service = WhatsAppService()
    results = []
    
    for phone_number in phone_numbers:
        result = service.send_message(phone_number, message)
        results.append({
            'phone_number': phone_number,
            'success': result['success'],
            'error': result.get('error')
        })
    
    return {
        'total_sent': len(phone_numbers),
        'results': results
    }


# Testing function
def test_whatsapp_integration():
    """Test WhatsApp integration"""
    service = WhatsAppService()
    
    # Test AI system
    test_questions = [
        "How do I get from Madina to Circle?",
        "What is a trotro?",
        "Where is Kaneshie station?",
        "Help me plan a trip from Kumasi to Accra"
    ]
    
    print("Testing AI responses:")
    for question in test_questions:
        result = service.mcp.process_message(question, 'whatsapp', 'test_user')
        print(f"\nQ: {question}")
        print(f"A: {result.get('response', 'No response')}")
    
    print("\n✅ WhatsApp integration test completed!")


if __name__ == "__main__":
    test_whatsapp_integration()
