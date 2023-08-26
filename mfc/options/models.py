from django.db import models
from solo.models import SingletonModel


# Create your models here.


class Options(SingletonModel):
    dataset = models.FileField(upload_to='ai_rubet_model/')
