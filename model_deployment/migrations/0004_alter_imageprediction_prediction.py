# Generated by Django 4.1.4 on 2023-01-10 07:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model_deployment", "0003_remove_prediction_prediction_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="imageprediction",
            name="prediction",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
