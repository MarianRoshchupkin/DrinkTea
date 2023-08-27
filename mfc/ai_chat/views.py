import json
import numpy as np
import pandas as pd
import random

from django.shortcuts import render, HttpResponse
from django.contrib.auth.decorators import user_passes_test
from django.http import JsonResponse

from options.models import Options
from .check import chek_question

from .models import QuestionAnswering
from .ai_model import calculate_similarity, convert_in_vector_massive

# QUESTIONS_IN_VECTOR_FORMAT = convert_in_vector_massive(list(
#     QuestionAnswering.objects.values_list('question_in_text_form', flat=True)))

test_list = ["вы ввели некорректный запрос", "я не отвечаю на такие запросы", "Я не могу реагировать на такие запросы.", "Некоторые запросы я не могу обработать, извините.", "К сожалению, не могу ответить на этот запрос в связи со своими ограничениями."]

def ai_chat(request):
    return render(request, "ai_chat/chat_demo.html")

def ai_chat_front(request):
    return render(request, "ai_chat/chat_front.html")

def ai_request(request):
    request_data = json.loads(request.body)
    if chek_question(request_data) == False:
        return JsonResponse({"response": random.choice(test_list)})
    else:

        print(chek_question(request_data))
        index = int(
            calculate_similarity(request_data,
                                 QuestionAnswering.objects.values_list('question_in_vector_form', flat=True)))
        response = QuestionAnswering.objects.values_list('answer_in_text_form', flat=True)[index]
        return JsonResponse({"response": response})

@user_passes_test(lambda user: user.is_superuser)
def ai_save_dataset(request):
    try:
        df = pd.read_excel(Options.get_solo().dataset)
        questions_in_text_form = list(map(str, df['QUESTION'].values))
        answers_in_text_form = list(map(str, df['ANSWER'].values))

        for question_in_text_form, answer_in_text_form in zip(questions_in_text_form, answers_in_text_form):
            question_in_vector_form = convert_in_vector_massive(np.array([question_in_text_form]))
            question_answering = QuestionAnswering(
                question_in_text_form=question_in_text_form,
                question_in_vector_form=question_in_vector_form,
                answer_in_text_form=answer_in_text_form
            )
            question_answering.save()

        global QUESTIONS_IN_VECTOR_FORMAT
        QUESTIONS_IN_VECTOR_FORMAT = convert_in_vector_massive(
            QuestionAnswering.objects.values_list('question_in_text_form', flat=True))
    except Exception as message:
        return HttpResponse(f'BAD {message}')
    return HttpResponse('GOOD')
