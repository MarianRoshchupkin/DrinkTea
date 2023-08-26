import json
import numpy as np
import pandas as pd

from django.shortcuts import render, HttpResponse
from django.contrib.auth.decorators import user_passes_test
from django.http import JsonResponse

from options.models import Options

from .models import QuestionAnswering
from .ai_model import calculate_similarity, convert_in_vector_massive

# QUESTIONS_IN_VECTOR_FORMAT = convert_in_vector_massive(list(
#     QuestionAnswering.objects.values_list('question_in_text_form', flat=True)))


def ai_chat(request):
    return render(request, "ai_chat/chat.html")


def ai_request(request):
    request_data = json.loads(request.body)
    index = int(
        calculate_similarity(request_data, QuestionAnswering.objects.values_list('question_in_vector_form', flat=True)))
    response = QuestionAnswering.objects.values_list('question_in_text_form', flat=True)[index]
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
