from django.urls import path
from .views import ChatbotView

urlpatterns = [
    path('analyze/', ChatbotView.as_view(), name='chatbot_analyze'),
    
]