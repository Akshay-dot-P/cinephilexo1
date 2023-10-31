from django import forms
from django.core.mail import send_mail
from django.shortcuts import render
from django.contrib.auth.models import User

from pymongo import MongoClient

from django import forms

class RegistrationForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)
    email = forms.EmailField()
    profile_pic = forms.ImageField(required=True)
    bio = forms.CharField(widget=forms.Textarea, required=False)

class ProfilePictureUpdateForm(forms.Form):
    profile_pic = forms.ImageField(required=False)

    
class LoginForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)   

class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(label='Email')
    username = forms.CharField(max_length=100)

class ResetPasswordForm(forms.Form):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)  

from django import forms
from .models import Review, RATE_CHOICES

class RateForm(forms.ModelForm):
    rate = forms.ChoiceField(choices=RATE_CHOICES, widget=forms.Select(), required=True)

    class Meta:
        model = Review
        fields = ['rate']

class ReviewForm(forms.ModelForm):

    text = forms.CharField(widget=forms.Textarea(attrs={'class': 'materialize-textarea'}), required=False)
    class Meta:
        model = Review
        fields = ['text']
        
   
        


   
