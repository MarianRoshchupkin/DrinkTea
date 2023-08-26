from django.shortcuts import render, HttpResponse


def ai_chat(request):
    return render(request, "ai_chat/chat.html")


def ai_request(request):
    ai_answer = {"ai_request работает"}
    return HttpResponse(ai_answer)
