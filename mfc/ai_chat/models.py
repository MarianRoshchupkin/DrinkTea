from django.db import models


class QuestionAnswering(models.Model):
    question = models.TextField(blank=False)
    answer = models.TextField(blank=False)