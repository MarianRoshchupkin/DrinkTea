import json

from django.shortcuts import render
from django.http import JsonResponse


def ai_chat(request):
    return render(request, "ai_chat/chat.html")


def ai_request(request):
    body = json.loads(request.body)
    ai_answer = {"response": f"ai_request работает {body}"}
    return JsonResponse(ai_answer)
