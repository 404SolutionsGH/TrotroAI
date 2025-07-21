from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging
from datetime import datetime
from .enhanced_ai_system import TrotroAI, TrotroMCP
from .whatsapp_integration import WhatsAppService

logger = logging.getLogger(__name__)

# Initialize AI system
ai_system = TrotroAI()
ai_system.load_model()
mcp = TrotroMCP(ai_system)

def ai_chat(request):
    """Web interface for AI chat"""
    return render(request, 'ai/chat.html')

@csrf_exempt
@require_http_methods(["POST"])
def ai_chat_api(request):
    """API endpoint for AI chat"""
    try:
        data = json.loads(request.body)
        question = data.get('question', '')
        session_id = data.get('session_id', 'web_session')
        
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        # Process question
        result = mcp.process_message(question, 'web', session_id)
        
        return JsonResponse({
            'success': True,
            'response': result.get('response', ''),
            'confidence': result.get('confidence', 0),
            'context': result.get('context', ''),
            'type': result.get('type', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Error in AI chat API: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET", "POST"])
def whatsapp_webhook(request):
    """WhatsApp webhook endpoint"""
    try:
        whatsapp_service = WhatsAppService()
        return whatsapp_service.handle_webhook(request)
        
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def send_whatsapp_message(request):
    """Send WhatsApp message"""
    try:
        data = json.loads(request.body)
        phone_number = data.get('phone_number', '')
        message = data.get('message', '')
        
        if not phone_number or not message:
            return JsonResponse({'error': 'Phone number and message are required'}, status=400)
        
        whatsapp_service = WhatsAppService()
        result = whatsapp_service.send_message(phone_number, message)
        
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@staff_member_required
@csrf_exempt
@require_http_methods(["POST"])
def train_model(request):
    """Train AI model with new data"""
    try:
        data = json.loads(request.body)
        additional_qa = data.get('qa_pairs', [])
        
        if not additional_qa:
            return JsonResponse({'error': 'No Q&A pairs provided'}, status=400)
        
        # Validate Q&A pairs
        for qa in additional_qa:
            if not qa.get('question') or not qa.get('answer'):
                return JsonResponse({'error': 'Each Q&A pair must have question and answer'}, status=400)
        
        # Train model
        ai_system.fine_tune_model(additional_qa)
        
        return JsonResponse({
            'success': True,
            'message': f'Model trained with {len(additional_qa)} new Q&A pairs',
            'total_qa_pairs': len(ai_system.sample_qa)
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@staff_member_required
def export_model(request):
    """Export AI model"""
    try:
        export_file = ai_system.export_model()
        
        return JsonResponse({
            'success': True,
            'export_file': export_file,
            'message': 'Model exported successfully'
        })
        
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def model_status(request):
    """Get AI model status"""
    try:
        return JsonResponse({
            'success': True,
            'model_info': {
                'total_qa_pairs': len(ai_system.sample_qa),
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'deepseek_api_configured': bool(ai_system.deepseek_api_key),
                'last_updated': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@staff_member_required
def admin_dashboard(request):
    """Admin dashboard for AI system"""
    try:
        context = {
            'total_qa_pairs': len(ai_system.sample_qa),
            'model_info': {
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'deepseek_api_configured': bool(ai_system.deepseek_api_key),
            }
        }
        return render(request, 'ai/admin_dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Error in admin dashboard: {e}")
        return render(request, 'ai/error.html', {'error': str(e)})

@staff_member_required
def analytics(request):
    """AI analytics dashboard"""
    try:
        # Basic analytics - you can expand this
        analytics_data = {
            'total_questions': len(ai_system.sample_qa),
            'question_types': {},
            'confidence_distribution': {},
        }
        
        # Count question types
        for qa in ai_system.sample_qa:
            qa_type = qa.get('type', 'unknown')
            analytics_data['question_types'][qa_type] = analytics_data['question_types'].get(qa_type, 0) + 1
        
        return JsonResponse({
            'success': True,
            'analytics': analytics_data
        })
        
    except Exception as e:
        logger.error(f"Error in analytics: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def answer_question_api(request):
    """API endpoint to answer questions"""
    try:
        data = json.loads(request.body)
        question = data.get('question', '')
        use_api = data.get('use_api', False)
        
        if not question:
            return JsonResponse({'error': 'Question is required'}, status=400)
        
        # Answer question
        result = ai_system.answer_question(question, use_api=use_api)
        
        return JsonResponse({
            'success': True,
            'question': result['question'],
            'answer': result['answer'],
            'confidence': result['confidence'],
            'context': result['context'],
            'type': result['type']
        })
        
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def get_suggestions(request):
    """Get question suggestions"""
    try:
        suggestions = [
            "How do I get from Madina to Circle?",
            "What is a trotro?",
            "Where is Kaneshie station?",
            "What's the fare from Kumasi to Accra?",
            "How do I find the nearest station?",
            "What cities does TrotroLive cover?",
            "Tell me about route 37",
            "What are the transport options in Ghana?",
            "How do I pay for trotro?",
            "What's the best way to travel from Osu to Tema?"
        ]
        
        return JsonResponse({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def submit_feedback(request):
    """Submit feedback for AI responses"""
    try:
        data = json.loads(request.body)
        question = data.get('question', '')
        answer = data.get('answer', '')
        rating = data.get('rating', 0)
        feedback = data.get('feedback', '')
        
        # Log feedback (you can store this in database)
        logger.info(f"Feedback received - Question: {question}, Rating: {rating}, Feedback: {feedback}")
        
        # You can implement feedback storage and model improvement here
        
        return JsonResponse({
            'success': True,
            'message': 'Thank you for your feedback!'
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return JsonResponse({'error': str(e)}, status=500)
