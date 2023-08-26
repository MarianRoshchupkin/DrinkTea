from django.shortcuts import render, HttpResponse


def ai_chat(request):
    return render(request, "chat.html")


def ai_request(request):
    ai_answer = None
    return HttpResponse(ai_answer)
