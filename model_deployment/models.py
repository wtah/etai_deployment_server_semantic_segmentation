from django.db import models

# Create your models here.
class Prediction(models.Model):
    created = models.DateTimeField(auto_now_add=True)


class TextPrediction(Prediction):
    sample = models.TextField()
    prediction = models.JSONField(blank=True, null=True,)

class ImagePrediction(Prediction):
    sample = models.ImageField(upload_to='upload')
    prediction = models.JSONField(null=True, blank=True)