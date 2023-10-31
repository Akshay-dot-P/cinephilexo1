from django.db import models
from django.contrib.auth.models import User
import uuid
import random

from django.db import models
from django.core.validators import RegexValidator

RATE_CHOICES = [
    (1, '1 - Trash'),
    (2, '2 - Horrible'),
    (3, '3 - Terrible'),
    (4, '4 - Bad'),
    (5, '5 - OK'),
    (6, '6 - Watchable'),
    (7, '7 - Good'),
    (8, '8 - Very Good'),
    (9, '9 - Perfect'),
    (10, '10 - Master Piece'), 
]


import random





class NumericAutoField(models.IntegerField):
    def save(self, *args, **kwargs):
        if not self.imdbID:
            self.imdbID = str(random.randint(100000, 999999))
        super().save(*args, **kwargs)
    # Add other fields as per your requirements

   
from django.core.validators import RegexValidator

    
class Movie(models.Model):
  
    # Define your movie fields here
    title = models.CharField(max_length=255)
    director = models.CharField(max_length=100)
    year = models.IntegerField()
    thumbnail = models.ImageField(upload_to='movie_thumbnails')
    thumbnailbg = models.ImageField(upload_to='thumbnailbg')
    genre = models.CharField(max_length=100)
    description = models.TextField()
    imdbID = models.CharField(max_length=10, unique=True, validators=[RegexValidator(r'^[0-9]+$')])

    # Other fields in your model





class Review(models.Model):
    user = models.CharField(max_length=150)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)
    text = models.TextField(max_length=3000, blank=True)
    rate = models.PositiveSmallIntegerField(choices=RATE_CHOICES)
    likes = models.PositiveIntegerField(default=0)
    unlikes = models.PositiveIntegerField(default=0)



  

class Watchlist(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    movies = models.ManyToManyField(Movie)

    def __str__(self):
        return f"Watchlist for {self.user.username}"   


