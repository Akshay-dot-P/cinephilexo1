# Generated by Django 3.2.6 on 2023-06-12 11:47

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('register', '0013_auto_20230612_1650'),
    ]

    operations = [
        migrations.AddField(
            model_name='movie',
            name='description',
            field=models.TextField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='movie',
            name='director',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='movie',
            name='genre',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='movie',
            name='thumbnail',
            field=models.ImageField(default=1, upload_to='movie_thumbnails'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='movie',
            name='thumbnailbg',
            field=models.ImageField(default=11, upload_to='thumbnailbg'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='movie',
            name='year',
            field=models.IntegerField(default=11),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='movie',
            name='imdbID',
            field=models.UUIDField(default=uuid.uuid4, editable=False, unique=True),
        ),
    ]
