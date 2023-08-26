from django.contrib import admin
from django.urls import path

from ai_chat.views import ai_chat, ai_request

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ai_chat/', ai_chat),
    path('ai_request/', ai_request, name="ai_request")
]
