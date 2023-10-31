


from .models import *

from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin

from .models import Movie, Review, Watchlist

# Define your custom admin class for User model
class CustomUserAdmin(UserAdmin):
    # Customize the list of displayed fields in the admin list view
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')

    # Add any other customizations you want, such as search fields or filters

# Unregister the default User admin
admin.site.unregister(User)

# Register the User model with your custom admin class
admin.site.register(User, CustomUserAdmin)

# Register other models
admin.site.register(Movie)
admin.site.register(Review)
admin.site.register(Watchlist)
