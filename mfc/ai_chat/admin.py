from django.contrib import admin
from .models import QuestionAnswering


# Register your models here.

class QuestionAnsweringAdmin(admin.ModelAdmin):
    list_display = ('question', 'answer')
    search_fields = ('question',)


admin.site.register(QuestionAnswering, QuestionAnsweringAdmin)
