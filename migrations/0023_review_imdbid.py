# Generated by Django 3.2.6 on 2023-07-19 12:37

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0022_watchlist'),
    ]

    operations = [
        migrations.AddField(
            model_name='review',
            name='imdbID',
            field=models.CharField(default=2, max_length=10, unique=True, validators=[django.core.validators.RegexValidator('^[0-9]+$')]),
            preserve_default=False,
        ),
    ]
