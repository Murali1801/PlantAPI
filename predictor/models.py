from django.db import models

# Create your models here.

class PredictionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    model_name = models.CharField(max_length=50)
    image_name = models.CharField(max_length=255)
    predicted_class = models.CharField(max_length=100)
    client_ip = models.GenericIPAddressField(null=True, blank=True)

    def __str__(self):
        return f"{self.timestamp} - {self.model_name} - {self.predicted_class}"
