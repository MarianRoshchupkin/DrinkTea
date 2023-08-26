from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from ai_chat.views import ai_chat, ai_request, ai_save_dataset, ai_chat_front

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ai_chat/', ai_chat),
    path('ai_chat_front/', ai_chat_front),
    path('ai_request/', ai_request, name="ai_request"),
    path('ai_save_dataset/', ai_save_dataset, name="ai_save_dataset")
]
