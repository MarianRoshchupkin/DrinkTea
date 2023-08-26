from django.contrib import admin
from .models import QuestionAnswering


# Register your models here.

class QuestionAnsweringAdmin(admin.ModelAdmin):
    list_display = ('question_in_text_form', 'answer_in_text_form')
    search_fields = ('question_in_text_form',)


admin.site.register(QuestionAnswering, QuestionAnsweringAdmin)
