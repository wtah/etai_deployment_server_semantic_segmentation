# Generated by Django 4.1.4 on 2023-01-10 06:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model_deployment", "0002_alter_imageprediction_sample"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="prediction",
            name="prediction",
        ),
        migrations.AddField(
            model_name="imageprediction",
            name="prediction",
            field=models.ImageField(blank=True, null=True, upload_to="preds"),
        ),
        migrations.AddField(
            model_name="textprediction",
            name="prediction",
            field=models.JSONField(blank=True, null=True),
        ),
    ]