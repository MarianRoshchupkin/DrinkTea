import numpy as np

from django.db import models
from ndarraydjango.fields import NDArrayField


class QuestionAnswering(models.Model):
    question_in_text_form = models.TextField(blank=False)
    question_in_vector_form = NDArrayField(dtype=np.float32)
    answer_in_text_form = models.TextField(blank=False)
