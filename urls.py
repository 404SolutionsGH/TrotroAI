from django.urls import path
from . import views

app_name = 'ai'

urlpatterns = [
    # AI Chat endpoints
    path('chat/', views.ai_chat, name='ai_chat'),
    path('chat/api/', views.ai_chat_api, name='ai_chat_api'),
    
    # WhatsApp Integration
    path('whatsapp/webhook/', views.whatsapp_webhook, name='whatsapp_webhook'),
    path('whatsapp/send/', views.send_whatsapp_message, name='send_whatsapp'),
    
    # Model management
    path('model/train/', views.train_model, name='train_model'),
    path('model/export/', views.export_model, name='export_model'),
    path('model/status/', views.model_status, name='model_status'),
    
    # Admin endpoints
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin/analytics/', views.analytics, name='analytics'),
    
    # API endpoints
    path('api/question/', views.answer_question_api, name='answer_question'),
    path('api/suggestions/', views.get_suggestions, name='get_suggestions'),
    path('api/feedback/', views.submit_feedback, name='submit_feedback'),
]
