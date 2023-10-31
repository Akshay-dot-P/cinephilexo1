# Generated by Django 3.2.6 on 2023-06-12 13:36

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0014_auto_20230612_1717'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movie',
            name='imdbID',
            field=models.CharField(default=uuid.uuid4, max_length=36, unique=True),
        ),
    ]
