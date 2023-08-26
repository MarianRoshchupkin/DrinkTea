from django.db import models

# Create your models here.

class QuestionAnswering(models.Model):
    question = models.TextField(blank=False)
    answer = models.TextField(blank=False)