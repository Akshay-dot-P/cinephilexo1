# Generated by Django 3.2.6 on 2023-07-19 13:11

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0023_review_imdbid'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='review',
            name='imdbID',
        ),
    ]
