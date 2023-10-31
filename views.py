from django.shortcuts import render
from .forms import RegistrationForm
from pymongo import MongoClient
from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.models import User
from .forms import ForgotPasswordForm
from django.urls import reverse
from django.shortcuts import render
from django.utils.crypto import get_random_string


from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_text







from django.conf import settings

from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import RegistrationForm  # Import your RegistrationForm
from pymongo import MongoClient
from django.conf import settings
import os



from .forms import LoginForm
import re
from django.contrib.auth.hashers import make_password, check_password


def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract form data
            print(form.cleaned_data)

            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            email = form.cleaned_data['email']
            bio = form.cleaned_data['bio']
            profile_pic = form.cleaned_data['profile_pic']
            hashed_password = make_password(password)

            # Connect to MongoDB
            client = MongoClient("mongodb://localhost:27017")
            db = client["regis"]
            collection = db["regis1"]
            
            # Check if username or email already exists
            existing_user = collection.find_one({"$or": [{"username": username}, {"email": email}]})
            
            if existing_user:
                # Username or email already exists, display an error message
                error_message = "Username or email already exists. Please choose a different username or email."
                return render(request, 'registration.html', {'form': form, 'error_message': error_message})
            
            # Check if the password meets the criteria
            if not re.search(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$', password):
                # Password doesn't meet the criteria, display an error message
                error_message = "Password must contain at least 8 characters including one uppercase letter, one lowercase letter, one digit, and one special symbol."
                return render(request, 'registration.html', {'form': form, 'error_message': error_message})
            
            # Create a new document
            data = {
                "username": username,
                "password": hashed_password,
                "email": email,
                "bio": bio,
                "profile_pic": profile_pic.name if profile_pic else None  # Store the file name
            }
            
            # Insert the document into the collection
            collection.insert_one(data)
            if profile_pic:
                profile_pic_path = f"{settings.MEDIA_ROOT}/{profile_pic.name}"
                with open(profile_pic_path, 'wb+') as destination:
                    for chunk in profile_pic.chunks():
                        destination.write(chunk)
                        # Get the profile picture of the logged-in user
            logged_in_user = request.user
            logged_in_user_profile_pic = logged_in_user.userprofile.profile_pic.url if hasattr(logged_in_user, 'userprofile') else None

            return render(request, 'registration_success.html')
    else:
        form = RegistrationForm()
    
    return render(request, 'registration.html', {'form': form})




from .forms import LoginForm


from django.shortcuts import render


from .forms import RegistrationForm
from pymongo import MongoClient
import re
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.shortcuts import render, redirect
from pymongo import MongoClient
from django.conf import settings
from .forms import RegistrationForm
from django.contrib.auth.decorators import login_required

@login_required
def update_profile(request):
    logged_in_user = request.session.get('username')

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]

    if request.method == 'POST':
        form = RegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            print(form.cleaned_data)

            # Extract form data
            profile_pic = form.cleaned_data['profile_pic']

            # Update the user's profile data
            username = logged_in_user

            if profile_pic:
                profile_pic_name = profile_pic.name
                print("Uploaded file name:", profile_pic_name)  # Print the uploaded file name


                # Save the uploaded picture to the media directory
                profile_pic_path = f"{settings.MEDIA_ROOT}/{profile_pic_name}"
                with open(profile_pic_path, 'wb+') as destination:
                    for chunk in profile_pic.chunks():
                        destination.write(chunk)

                # Update profile picture URL in MongoDB collection
                collection.update_one({"username": username}, {"$set": {"profile_pic": profile_pic_name}})

                # Redirect to the profile page after updating
                return redirect('profile')

    else:
        form = RegistrationForm()

    context = {
        "logged_in_user": logged_in_user,
        "form": form,  # Include the form in the context
        # Add other profile-related context variables
    }

    return render(request, 'profile.html', context)








def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            # Extract form data
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            # Connect to MongoDB
            client = MongoClient("mongodb://localhost:27017")
            db = client["regis"]
            collection = db["regis1"]
            logged_in_user = request.session.get('username')

            # Connect to MongoDB
            
            # Retrieve the logged-in user's profile information
            logged_in_user_data = collection.find_one({"username": logged_in_user})

            logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

            if logged_in_user_profile_pic:
                logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
            else:
                logged_in_user_profile_pic_url = None

            # Perform login authentication
            user = collection.find_one({"username": username})

            if user and check_password(password, user['password']):
                # Login successful
                
                # Retrieve the user document from MongoDB
                user_doc = collection.find_one({"username": username})
                
                if user_doc:
                    # User found in MongoDB, extract the username
                    username = user_doc["username"]
                    request.session['username'] = username

                    username = request.session.get('username')
                    print(username)
                    # Connect to MongoDB
                    client = MongoClient("mongodb://localhost:27017")
                    db = client["regis"]
                    ratings_collection = db["ratings"]

                    # Check if the user exists in the ratings collection
                    if ratings_collection.find_one({"user": username}):
                        
                        # Retrieve all ratings data from MongoDB
                        ratings_data = list(ratings_collection.find())
                        print(ratings_data)

                        # Create a DataFrame from the ratings data
                        ratings_df = pd.DataFrame(ratings_data)
                        ratings_df['rate'] = pd.to_numeric(ratings_df['rate'])
                        print(ratings_df['rate'])

                        if not ratings_df.empty:
                            # Create a user-item matrix with users as rows and movies as columns
                            user_item_matrix = ratings_df.pivot(index='user', columns='movie', values='rate').fillna(0)
                            print("user_item_matrix")
                            print(user_item_matrix)

                            # Calculate item similarity based on cosine similarity
                            item_similarity_matrix = cosine_similarity(user_item_matrix.T)
                            print("item_similarity_matrix")
                            print(item_similarity_matrix)

                            # Convert the similarity matrix into a DataFrame
                            item_similarity_df = pd.DataFrame(item_similarity_matrix, columns=user_item_matrix.columns, index=user_item_matrix.columns)
                            print("item_similarity_df")
                            print(item_similarity_df)

                            # Get the user's rated movies
                            user_rated_movies = user_item_matrix.loc[username].index[user_item_matrix.loc[username] > 0]
                            print("user rated movies")
                            print(user_rated_movies)

                            # Get similar movies for each user-rated movie
                            recommended_movies = []
                            for movie in user_rated_movies:
                                similar_movies = item_similarity_df[movie].sort_values(ascending=False)
                                recommended_movies.extend(similar_movies.index[1:])  # Exclude the current movie
                            print("recommended movies before removing duplicates and movies the user has already rated")    
                            print(recommended_movies)
                            # Remove duplicates and movies the user has already rated
                            #recommended_movies = [movie for movie in recommended_movies if movie not in user_rated_movies]
                            print("recommended movies after  removing duplicates and movies the user has already rated")    
                            print(recommended_movies)
                            # Get movie objects for recommended movie titles
                            recommended_movie_objects = Movie.objects.filter(title__in=recommended_movies)
                            print(recommended_movie_objects)

                        else:







                            





                            # Handle the case where there are no ratings
                            recommended_movie_objects = top_rated_movies
                            context = {
                            "username": username,
                            "logged_in_user": logged_in_user,
                            "logged_in_user_profile_pic_url": logged_in_user_profile_pic_url,
                            "recommended_movies": recommended_movie_objects,
                            }
                            print(recommended_movie_objects)

                        # Pass the username to the template
                        context = {
                            "username": username,
                            "logged_in_user": logged_in_user,
                            "logged_in_user_profile_pic_url": logged_in_user_profile_pic_url,
                            "recommended_movies": recommended_movie_objects,
                        }
                        return render(request, 'login_success.html', context)
                    else:
                        client = MongoClient("mongodb://localhost:27017")
                        db = client['regis']
                        collection = db['register_movie']
                        ratings_collection = db['ratings']

                        movies = Movie.objects.all()

                        for movie in movies:
                            ratings = list(ratings_collection.find({"movie": movie.title}))
                            all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
                            avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
                            movie.avg_rating = avg_rating

                        top_rated_movies = sorted(
                        movies,
                        key=lambda x: x.avg_rating if x.avg_rating is not None else float('-inf'),
                        reverse=True
                        )[:10]
                        context = {
                            "username": username,
                            "logged_in_user": logged_in_user,
                            "logged_in_user_profile_pic_url": logged_in_user_profile_pic_url,
                            "recommended_movies": top_rated_movies,

                        }
                        return render(request, 'login_success.html', context)
                        # User not found in the ratings collection

                else:
                    # User not found in MongoDB
                    error_message = "User data not found."
                    return render(request, 'login.html', {'form': form, 'error_message': error_message})
            else:
                # Login failed
                error_message = "Invalid username or password. Please try again."
                return render(request, 'login.html', {'form': form, 'error_message': error_message})
                
    else:
        form = LoginForm()
    
    return render(request, 'login.html', {'form': form})






from django.shortcuts import render
from .forms import ForgotPasswordForm
from pymongo import MongoClient
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail

def forgot_password(request):
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            # Extract form data
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            
            # Connect to MongoDB
            client = MongoClient("mongodb://localhost:27017")
            db = client["regis"]
            collection = db["regis1"]
        
            # Check if email exists in the database
            user = collection.find_one({"username": username, "email": email})
            
            if user:
                # Generate password reset token and URL
                token = get_random_string(length=20)  # Generate a random token
                collection.update_one({"email": email}, {"$set": {"reset_token": token}})
              
                reset_url = request.build_absolute_uri(
                    f'/reset-password/?email={email}&token={token}'
                )
                
                # Send password reset email
                send_password_reset_email(email, reset_url)
                
                return render(request, 'password_reset_request_success.html')
            else:
                # Email not found in the database
                error_message = "No user found with the provided email address."
                return render(request, 'forgot_password.html', {'form': form, 'error_message': error_message})
    else:
        form = ForgotPasswordForm()
    
    return render(request, 'forgot_password.html', {'form': form})

def send_password_reset_email(email, reset_url):
    subject = "Password Reset Request"
    message = f"Click the following link to reset your password: {reset_url}"
    sender = "akshayp7841@gmail.com"  # Replace with your email address
    recipient = email

    send_mail(subject, message, sender, [recipient])



from django.shortcuts import render, redirect
from .forms import ResetPasswordForm
from pymongo import MongoClient

def reset_password(request):
    if request.method == 'POST':
        form = ResetPasswordForm(request.POST)
        if form.is_valid():
            # Extract form data
            password = form.cleaned_data['password']
            confirm_password = form.cleaned_data['confirm_password']
            
            # Perform password reset logic
            if password == confirm_password:
                # Retrieve email and token from the reset URL parameters
                email = request.GET.get('email')
                token = request.GET.get('token')
                
                # Connect to MongoDB and update user's password
                client = MongoClient("mongodb://localhost:27017")
                db = client["regis"]
                collection = db["regis1"]
                
                # Find the user in the database using the email and token
                user = collection.find_one({"email": email, "reset_token": token})
                
                if user:
                    # Update the user's password field in the database
                    # You'll need to adjust the field name according to your database schema
                    collection.update_many({"email": email, "reset_token": token}, {"$set": {"password": password}})
                    
                    # Invalidate the token (optional)
                    # You can remove the token from the user document or mark it as expired
                    # depending on your specific requirements
                    
                    # Provide feedback to the user about the success of the password reset
                    success_message = "Your password has been successfully reset."
                    return render(request, 'reset_password_success.html', {'success_message': success_message})
                else:
                    # Invalid email or token, display an error message
                    error_message = "Invalid email or token."
                    return render(request, 'reset_password.html', {'form': form, 'error_message': error_message})
            else:
                # Passwords don't match, display an error message
                error_message = "Passwords do not match."
                return render(request, 'reset_password.html', {'form': form, 'error_message': error_message})
    else:
        form = ResetPasswordForm()
    
    return render(request, 'reset_password.html', {'form': form})





from django.shortcuts import get_object_or_404



def movies(request):
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]

    
    movies = Movie.objects.all()
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]
    logged_in_user = request.session.get('username')

    logged_in_user_data = collection.find_one({"username": logged_in_user})

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None
    context = {
        'movies': movies,
        'logged_in_user_profile_pic_url': logged_in_user_profile_pic_url,

    }

    collection = db["register_movie"]
    ratings_collection = db["ratings"]


    for movie in movies:
        ratings = list(ratings_collection.find({"movie": movie.title}))
        all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats

        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
        movie.avg_rating = avg_rating
    return render(request, 'movies.html', context)

    
from pymongo import MongoClient
from django.shortcuts import render, get_object_or_404, redirect
import statistics
from .forms import ReviewForm
from bson import ObjectId

from django.http import JsonResponse


# Other import statements...

from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404, render
from .models import Movie
from .forms import ReviewForm
from pymongo import MongoClient
from bson import ObjectId
import json
import statistics
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  # Import numpy for numeric operations
from django.db.models import Q
def movie_details(request, imdb_id):
    # Connect to MongoDB and retrieve movie data
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]
    movie = get_object_or_404(Movie, imdbID=imdb_id)
    selected_movie = get_object_or_404(Movie, imdbID=imdb_id)

    # Get the genre of the selected movie
    selected_genre = selected_movie.genre

    # Retrieve movies with the same genre but exclude the selected movie
    recommended_movies = Movie.objects.filter(genre=selected_genre).exclude(imdbID=imdb_id)

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.cleaned_data['text']
            username = request.session.get('username')
            username1 = request.user.username if request.user.is_authenticated else None

            # Check if the user has already submitted a review
            reviews_collection = db["reviews"]
            existing_review = reviews_collection.find_one({"user": username, "movie": movie.title})

            if existing_review:
                # User has already submitted a review
                response_data = {
                    'message': "You have already submitted a review for this movie."
                }
                return JsonResponse(response_data, status=400)  # Return a bad request response with the message
            review_id = str(ObjectId())

            # Save the review to the database

            review_data = {
                'user': username,
                'movie': movie.title,
                '_id': review_id,  # Store the MongoDB ObjectId in the _id field
                'id': review_id,
                'text': review,
                'imdbID': imdb_id,
            }
            reviews_collection.insert_one(review_data)

            # Construct the response data
            response_data = {
                'my_review': review,
            }

            # Redirect back to movie details page
            return JsonResponse(response_data)

    elif request.method == 'PUT':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        existing_review = reviews_collection.find_one({'id': ObjectId(review_id)})

        if existing_review:
            data = json.loads(request.body.decode('utf-8'))
            updated_review_text = data.get('text', '')

            # Update the review in the MongoDB collection using update_one
            result = reviews_collection.update_one(
                {'id': ObjectId(review_id)},
                {'$set': {'text': updated_review_text}}
            )

            if result.modified_count > 0:
                # Return a success response if the document was updated successfully
                return JsonResponse({'message': 'Review updated successfully'})
            else:
                # Return an error response if the document was not updated
                return JsonResponse({'error': 'Failed to update review'}, status=400)
        else:
            return JsonResponse({'message': 'Review not found'}, status=404)

    elif request.method == 'DELETE':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        result = reviews_collection.delete_one({'id': ObjectId(review_id)})

        if result.deleted_count > 0:
            # Return a success response if the document was deleted successfully
            return HttpResponse(status=204)  # Return a success response with no content
        else:
            # Return an error response if the document was not found or not deleted
            return JsonResponse({'error': 'Review not found'}, status=404)
    
    username = request.session.get('username')

    # Query the existing reviews for the movie
    reviews_collection = db["reviews"]
    reviews = list(reviews_collection.find({"movie": movie.title}))

    # Query the existing ratings
    ratings_collection = db["ratings"]
    ratings = list(ratings_collection.find({"user": username, "movie": movie.title}))

    # Calculate the average rating for the movie
    all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
    avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None

    # Retrieve the username from the session
    username = request.session.get('username')

    # Find the user's own rating
    my_rating = None
    for rating in ratings:
        if rating['user'] == username:
            my_rating = rating['rate']
            break

    my_review = None
    for review in reviews:
        if review['user'] == username:
            my_review = review['text']
            break

    # Specify the database and collection
    collection_name = "ratings"  # Replace with your ratings collection name

    # Retrieve data from the MongoDB collection
    ratings_collection = db["ratings"]
    ratings_data = list(ratings_collection.find({"movie": movie.title}))
    user_ratings = list(ratings_collection.find({"user": username}))

    # Create a DataFrame from the retrieved data
    ratings_df = pd.DataFrame(ratings_data)
    print(user_ratings)
    if 'rate' in ratings_df:
        all_ratings = [float(rate) for rate in ratings_df['rate']]  # Convert strings to floats
        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
    
    # Calculate the average rating for the movie

    # Retrieve the username from the session
        username = request.session.get('username')

    # Find the user's own rating

    # Continue with your collaborative filtering logic using the ratings_df DataFrame
        if my_rating is not None:
            username = request.session.get('username')

            # Connect to MongoDB
            client = MongoClient("mongodb://localhost:27017")
            db = client["regis"]
            ratings_collection = db["ratings"]

            # Retrieve all ratings data from MongoDB
            ratings_data = list(ratings_collection.find())

            # Create a DataFrame from the ratings data
            ratings_df = pd.DataFrame(ratings_data)

            print("Ratings Data:")
            print(ratings_df)

            if not ratings_df.empty:
                # Create a user-item matrix with users as rows and movies as columns
                user_item_matrix = ratings_df.pivot(index='user', columns='movie', values='rate').fillna(0)

                print("User-Item Matrix:")
                print(user_item_matrix)

                # Calculate user similarity based on cosine similarity
                user_similarity_matrix = cosine_similarity(user_item_matrix)

                print("User Similarity Matrix:")
                print(user_similarity_matrix)
                print(username)

                # Get the ratings of the logged-in user
                if username in user_item_matrix.index:
                    user_ratings = user_item_matrix.loc[username]
                    print(user_ratings)
                    # You can set it to None or handle it as needed

                    # Find similar users by sorting the user similarity matrix
                    similar_users_indices10 = np.argsort(-user_similarity_matrix[user_item_matrix.index == username])
                    print(similar_users_indices10)
                    num_users = user_item_matrix.shape[0]  # Total number of users
                    percentage_to_select = 90  # Change this to your desired percentage
                    N = int(num_users * (percentage_to_select / 100))

                    # Extract the indices of similar users excluding the current user
                    similar_user_indices1 = similar_users_indices10[0]
                    similar_user_indices = similar_user_indices1[:N]

                    similar_user_indices = similar_user_indices1[similar_user_indices1 != user_item_matrix.index.get_loc(username)]
                    print(similar_user_indices1)
                    user_names = user_item_matrix.index.to_numpy()
                    

                    user_at_index_1 = user_names[similar_user_indices1[0]]
                    user_at_index_2 = user_names[similar_user_indices1[1]]

                    print("User at index 1:", user_at_index_1)
                    print("User at index 2:", user_at_index_2)

                    # Get the names of similar users
                    user_names = user_item_matrix.index.to_numpy()
                    similar_user_names = user_names[similar_user_indices]
                    print(similar_user_names)

                    # Now similar_user_names contains the names of users similar to the logged-in user, including the current user

                    # Filter the user-item matrix to include only similar users
                    # Create a boolean mask to filter the user-item matrix
                    similar_users_mask = user_item_matrix.index.isin(similar_user_names)

                    # Apply the mask to get the user-item matrix for similar users
                    similar_user_item_matrix = user_item_matrix[similar_users_mask]
                    print("similar_user_item_matrix")
                    print(similar_user_item_matrix)
                    similar_user_item_matrix1 = similar_user_item_matrix.astype(float)

                    # Calculate the mean ratings for each movie
                    recommended_movie_ratings = similar_user_item_matrix1.mean()
                    recommended_movie_ratings = recommended_movie_ratings.apply(str)

                    print(recommended_movie_ratings)

                    # Filter out movies the user has already rated

                    # Sort recommended movies by rating in descending order
                    recommended_movie_ratings = recommended_movie_ratings.sort_values(ascending=False)
                    selected_movie = get_object_or_404(Movie, imdbID=imdb_id)
                    print("selected_movie:", selected_movie)
                    print("Top Recommended Movie Ratings:")
                    print(recommended_movie_ratings)

                    # Get the top N recommended movies
                    top_n_recommendations = recommended_movie_ratings.head(10)

                    print("Top N Recommended Movies:")
                    print(top_n_recommendations)

                    # Fetch movie objects for recommended movie titles
                    # Create a dictionary mapping movie titles to their ratings
                    movie_ratings_dict = dict(zip(recommended_movie_ratings.index, recommended_movie_ratings))

                    # Print the movie_ratings_dict to check its content
                    print(movie_ratings_dict)

                    # Fetch movie objects for recommended movie titles
                    recommended_movie_objects1 = Movie.objects.filter(title__in=top_n_recommendations.index)

                    # Print recommended_movie_objects before sorting
                    print("Recommended Movies before sorting:")
                    print(recommended_movie_objects1)

                    # Sort the recommended_movie_objects based on ratings (descending order)
                    # Sort the recommended_movie_objects based on ratings (descending order) and exclude the selected movie
                    recommended_movie_objects12 = Movie.objects.filter(title__in=top_n_recommendations.keys())[:7]


                    recommended_movie_objects = sorted(movie_ratings_dict.items(), key=lambda x: x[1], reverse=True)
                    

                    # Print recommended_movie_objects after sorting
                    print("Recommended Movies after sorting:")
                    print(recommended_movie_objects)
                else:
                    user_ratings = None  # You can set it to None or handle it as needed
            else:
                genre_filter = Q()
                for genre in selected_genre:
                    genre_filter |= Q(genre__contains=genre)

# Query the movies with matching genres, excluding the selected movie
                recommended_movie_objects12 = Movie.objects.filter(genre_filter).exclude(imdbID=imdb_id)[:7]
                # Handle the case where there are no ratings
        else:
            # Handle the case where the user has not provided a rating
            genre_filter = Q()
            for genre in selected_genre:
                genre_filter |= Q(genre__contains=genre)

# Query the movies with matching genres, excluding the selected movie
            recommended_movie_objects12 = Movie.objects.filter(genre_filter).exclude(imdbID=imdb_id)[:7]
    else:
        selected_genres = selected_movie.genre.split(', ')  # Split genres into a list if they are comma-separated
        print(selected_genres)

        genre_filter = Q()
        for genre in selected_genres:
            genre_filter |= Q(genre__contains=genre)

# Query the movies with matching genres, excluding the selected movie
        recommended_movie_objects12 = Movie.objects.filter(genre_filter).exclude(imdbID=imdb_id)[:7]
        print(recommended_movie_objects12)
    context = {
        'movie': movie,
        'reviews': reviews,  # Add reviews to the context
        'my_review': my_review,
        'ratings': ratings,
        'avg_rating': avg_rating,
        'my_rating': my_rating,
        'selected_movie': selected_movie,
        'recommended_movies': recommended_movie_objects12,
    }

    return render(request, 'movie_details.html', context)




from django.shortcuts import render, redirect
from .forms import RateForm
from .models import Movie, Review
from statistics import mean

def rate(request, imdb_id):
    movie = Movie.objects.get(imdbID=imdb_id)
    user = request.user

    if request.method == 'POST':
        form = RateForm(request.POST)
        if form.is_valid():
            RATE = FORM.Save(commit=False)
            rate.user = user
            rate.movie = movie
            rate.save()
            return HttpResponseRedirect(reverse('movie-details', args=[imdb_id]))
    else:
        form = RateForm()

    template_name = loader.get_template('rate.html')
    context = {
        'form': form,
        'movie': movie,
    }

    return HttpResponse(template.render(context, Request))


from django.shortcuts import render

def rate_movie_popup(request, imdb_id):
    # Retrieve the movie object based on the imdb_id
    movie = get_object_or_404(Movie, imdbID=imdb_id)

    context = {
        'movie': movie,
    }

    return render(request, 'rate_movie_popup.html', context)





from django.shortcuts import render, redirect
from pymongo import MongoClient
from django.conf import settings
from .forms import ProfilePictureUpdateForm  # Update this import based on your actual form class
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from pymongo import MongoClient
from django.conf import settings
from .forms import ProfilePictureUpdateForm
from django.contrib.auth.decorators import login_required

def profile(request):
    logged_in_user = request.session.get('username')

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]

    if request.method == 'POST':
        form = ProfilePictureUpdateForm(request.POST, request.FILES)
        if form.is_valid():
            # Extract form data
            profile_pic = form.cleaned_data['profile_pic']

            # Update the user's profile data
            username = request.session.get('username')

            # Check if the user exists in the MongoDB collection
            user_document = collection.find_one({"username": username})
            if user_document:
                if profile_pic:
                    profile_pic_name = profile_pic.name

                    # Save the uploaded picture to the media directory
                    profile_pic_path = f"{settings.MEDIA_ROOT}/{profile_pic_name}"
                    with open(profile_pic_path, 'wb+') as destination:
                        for chunk in profile_pic.chunks():
                            destination.write(chunk)

                    # Update profile picture URL in MongoDB collection
                    collection.update_one({"username": username}, {"$set": {"profile_pic": profile_pic_name}})

                    # Return the updated profile picture URL
                    response_data = {
                        'profile_pic_url': f"{settings.MEDIA_URL}{profile_pic_name}".replace('\\', '/')
                    }
                    return JsonResponse(response_data)
            else:
                # Handle the case where the user does not exist
                # You might want to return an error response
                pass
    else:
        form = ProfilePictureUpdateForm()

    logged_in_user_data = collection.find_one({"username": logged_in_user})
    email = logged_in_user_data["email"]
    request.session['email'] = email
 

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None

    context = {
        "logged_in_user": logged_in_user,
        "logged_in_user_profile_pic_url": logged_in_user_profile_pic_url,
        "email": email,
        "form": form,  # Include the form in the context
        # Add other profile-related context variables
    }

    return render(request, 'profile.html', context)


from django.shortcuts import render
from .models import Movie

def genre_categorize(request, genre):
    username = request.session.get('username')
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]
    logged_in_user = request.session.get('username')

    logged_in_user_data = collection.find_one({"username": logged_in_user})

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]
    
    movies = Movie.objects.filter(genre__icontains=genre)
    context = {
        'genre': genre,
        'movies': movies,
        'logged_in_user_profile_pic_url': logged_in_user_profile_pic_url,

    }
    return render(request, 'genre_categorize.html', context)













from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Movie, Review
import json
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
import json
from django.http import JsonResponse

@csrf_exempt
def edit_review(request, imdb_id):
    # Retrieve the user's review for the specified movie
    if request.method == 'GET':
        client = MongoClient("mongodb://localhost:27017")
        db = client["regis"]
        collection = db["register_movie"]
        reviews_collection = db["reviews"]
        
        movie = get_object_or_404(Movie, imdbID=imdb_id)
        username = request.session.get('username')
        my_review = reviews_collection.find_one({"user": username, "movie": movie.title})
        my_review_mongodb = reviews_collection.find_one({"user": username, "movie": movie.title})

        reviews = list(reviews_collection.find({"movie": movie.title}))

        my_review = my_review_mongodb['text'] if my_review_mongodb else None

        try:
            review = Review.objects.get(movie=movie, user=username)
            context = {'movie': movie, 'my_review': review.text}
            return render(request, 'edit_review.html', context)
        except Review.DoesNotExist:
            context = {'movie': movie, 'my_review': my_review}
            return render(request, 'edit_review.html', context)

    # Handle the review update when the form is submitted via AJAX
    elif request.method == 'POST' and request.is_ajax():


        form = ReviewForm(request.POST)
        if form.is_valid():


    # Fetch the review text from the JSON data
            data = json.loads(request.body.decode('utf-8'))
            review_text = data.get('text', '')            # Retrieve the username from the session
            username = request.session.get('username')
            print("Request Body:", request.body)

            # Check if the user already has a review for the movie
            client = MongoClient("mongodb://localhost:27017")
            db = client["regis"]
            reviews_collection = db["reviews"]
            movie = get_object_or_404(Movie, imdbID=imdb_id)

            existing_review = reviews_collection.find_one({"user": username, "imdbID": movie.imdbID})

            if existing_review:


                # Update the existing review
                reviews_collection.update_one({"user": username, "imdbID": movie.imdbID}, {"$set": {"text": review_text}})

            else:
                # Insert a new review
                review_data = {
                    'user': username,
                    'imdbID': movie.imdbID,
                    'text': review_text,
                }
                reviews_collection.insert_one(review_data)

            return JsonResponse({'message': 'Review updated successfully'}, status=200)

    else:
        form = ReviewForm()
        # Query the existing reviews
        reviews_collection = db["reviews"]
        reviews = list(reviews_collection.find({"imdbID": movie["imdbID"]}))
        context = {
            'movie': movie,
            'reviews': reviews,
            'form': form,
        }

    return render(request, 'update_review.html', context)




# views.py

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from pymongo import MongoClient
from bson.objectid import ObjectId

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
from pymongo import MongoClient
from bson.objectid import ObjectId

from django.shortcuts import get_object_or_404
from .models import Movie, Review



from .models import Movie
from pymongo import MongoClient

def delete_review(request, imdb_id):
    if request.method == 'POST' and request.is_ajax():
        # Retrieve the username from the session
        username = request.session.get('username')

        # Connect to the MongoDB collection
        client = MongoClient("mongodb://localhost:27017")
        db = client["regis"]
        reviews_collection = db["reviews"]

        # Find and delete the user's review for the specified movie
        movie = get_object_or_404(Movie, imdbID=imdb_id)
        reviews_collection.delete_one({"user": username, "imdbID": movie.imdbID})

        return JsonResponse({'message': 'Review deleted successfully'}, status=200)

    # If the request is not a POST or not an AJAX request, return a 405 Method Not Allowed status
    return JsonResponse({'error': 'Method not allowed'}, status=405)






# views.py












# views.py

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Movie, Review
import json
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
import json
from django.http import JsonResponse

@csrf_exempt
def vote_review(request):
    if request.method == 'POST' and request.is_ajax():
        data = json.loads(request.body.decode('utf-8'))
        imdb_id = data.get('imdb_id')
        username = data.get('username')
        upvotes = data.get('upvotes')
        downvotes = data.get('downvotes')
        user_voted = data.get('user_voted')

        client = MongoClient("mongodb://localhost:27017")
        db = client["regis"]
        reviews_collection = db["reviews"]

        # Update the user's vote status in the MongoDB collection
        review = reviews_collection.find_one({"user": username, "imdbID": imdb_id})
        if review:
            reviews_collection.update_one(
                {"user": username, "imdbID": imdb_id},
                {"$set": {"upvotes": upvotes, "downvotes": downvotes, "user_voted": user_voted}},
            )

        return JsonResponse(
            {
                "upvotes": upvotes,
                "downvotes": downvotes,
                "user_voted": user_voted,
            },
            status=200,
        )

    return JsonResponse({"message": "Invalid request"}, status=400)







from django.shortcuts import render
from .models import Movie
from pymongo import MongoClient

def search_movies(request):
    # Get the search query from the URL parameter 'q' and strip any leading/trailing whitespace
    search_query = request.GET.get('q', '').strip()

    # If the search query is not empty, perform the search
    if search_query:
        # Use case-insensitive filtering to find movies with titles containing the search query
        matching_movies = Movie.objects.filter(title__icontains=search_query)

        # If no matching movies are found, you can add a message to display in the template
        if not matching_movies:
            message = f'No results found for "{search_query}".'
        else:
            message = f'Search Results for "{search_query}"'
    else:
        # If the search query is empty, display all movies
        matching_movies = Movie.objects.all()
        message = 'All Movies'

    # Pass the matching_movies and message to the template
    context = {
        'movies': matching_movies,
        'message': message,
    }

    return render(request, 'search_results.html', context)





from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from pymongo import MongoClient
from bson import ObjectId
from .models import Movie
from .forms import RateForm
import statistics


def rate_movie(request, imdb_id):
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]
    movie = get_object_or_404(Movie, imdbID=imdb_id)

    if request.method == 'POST' and request.is_ajax():
        form = RateForm(request.POST)
        if form.is_valid():
            rating = form.cleaned_data['rate']
            
            # Retrieve the username from the session
            username = request.session.get('username')
            
            # Check if the user already rated the movie
            ratings_collection = db["ratings"]
            user_rating = ratings_collection.find_one({"user": username, "movie": movie.title})
            if user_rating:
                # Update the existing rating
                ratings_collection.update_one({"user": username, "movie": movie.title}, {"$set": {"rate": rating}})
            else:
                # Insert a new rating
                review_data = {
                    'user': username,
                    'movie': movie.title,
                    'rate': rating,
                }
                ratings_collection.insert_one(review_data)

            # Query the updated ratings after adding or updating the rating
            ratings = ratings_collection.find({"movie": movie.title})

            # Calculate the average rating
            all_ratings = [float(review['rate']) for review in ratings]
            avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
            
            # Retrieve the username from the session
            username = request.session.get('username')

            # Check if the user already rated the movie
            user_rating = ratings_collection.find_one({"user": username, "movie": movie.title})
            my_rating = user_rating['rate'] if user_rating else None

            # Construct the JSON response
            response = {
                'user': username,
                'rate': rating,
                'avg_rating': avg_rating,
                'my_rating': my_rating
            }
            
            return JsonResponse(response)

    else:
        form = RateForm()
        # Query the existing ratings
        ratings_collection = db["ratings"]
        ratings = list(ratings_collection.find({"movie": movie.title}))

    # Calculate the average rating for the movie
    all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
    avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None

    # Retrieve the username from the session
    username = request.session.get('username')

    # Check if the user already rated the movie
    user_rating = ratings_collection.find_one({"user": username, "movie": movie.title})
    my_rating = user_rating['rate'] if user_rating else None

    context = {
        'movie': movie,
        'ratings': ratings,
        'form': form,
        'avg_rating': avg_rating,
        'my_rating': my_rating
    }

    return render(request, 'rate_movie.html', context)










from django.shortcuts import render, redirect, get_object_or_404
from .models import Movie, Watchlist

# ...






from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Movie
from pymongo import MongoClient


def add_to_watchlist(request, movie_title):
    movie = get_object_or_404(Movie, title=movie_title)

    # Retrieve the username from the session
    username = request.session.get('username')

    # Add the movie to the user's watchlist in MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    watchlist_collection = db["watchlist"]
    user_watchlist = watchlist_collection.find_one({"user": username})

    movie_titles = user_watchlist.get("movies", []) if user_watchlist else []

    if movie_title in movie_titles:
        return JsonResponse({'message': 'Movie already exists in watchlist section.'})
    else:
        watchlist_collection.update_one({"user": username}, {"$addToSet": {"movies": movie_title}}, upsert=True)
        return JsonResponse({'message': 'Movie added to watchlist successfully.'})











      

from django.shortcuts import render
from pymongo import MongoClient
from .models import Movie
from django.core.exceptions import ObjectDoesNotExist

from django.shortcuts import render
from pymongo import MongoClient
import statistics

def watchlist(request):
    # Retrieve the username from the session
    username = request.session.get('username')
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]
    logged_in_user = request.session.get('username')

    logged_in_user_data = collection.find_one({"username": logged_in_user})

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None

    # Retrieve the user's watchlist from MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    watchlist_collection = db["watchlist"]
    user_watchlist = watchlist_collection.find_one({"user": username})
    
    # Get the list of movie titles from the user's watchlist
    movie_titles = user_watchlist.get("movies", []) if user_watchlist else []

    # Retrieve the movie details from the register_movie collection using the movie titles
    movies = Movie.objects.filter(title__in=movie_titles)

    # Calculate the average rating for each movie
    ratings_collection = db["ratings"]
    for movie in movies:
        ratings = list(ratings_collection.find({"movie": movie.title}))
        all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats

        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
        movie.avg_rating = avg_rating

    context = {
        'movies': movies,
        'logged_in_user_profile_pic_url': logged_in_user_profile_pic_url,

    }

    return render(request, 'watchlist.html', context)










from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Movie
from pymongo import MongoClient

def add_to_favlist(request, movie_title):
    movie = get_object_or_404(Movie, title=movie_title)

    # Retrieve the username from the session
    username = request.session.get('username')

    # Add the movie to the user's watchlist in MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    watchlist_collection = db["favlist"]
    user_watchlist = watchlist_collection.find_one({"user": username})

    movie_titles = user_watchlist.get("movies", []) if user_watchlist else []

    if movie_title in movie_titles:
        return JsonResponse({'message': 'Movie already exists in favlist section.'})
    else:
        watchlist_collection.update_one({"user": username}, {"$addToSet": {"movies": movie_title}}, upsert=True)
        return JsonResponse({'message': 'Movie added to favlist successfully.'})






      

from django.shortcuts import render
from pymongo import MongoClient
from .models import Movie
from django.core.exceptions import ObjectDoesNotExist

from django.shortcuts import render
from pymongo import MongoClient
import statistics

def favlist(request):
    # Retrieve the username from the session
    username = request.session.get('username')
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]
    logged_in_user = request.session.get('username')

    logged_in_user_data = collection.find_one({"username": logged_in_user})

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None
    # Retrieve the user's watchlist from MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    watchlist_collection = db["favlist"]
    user_watchlist = watchlist_collection.find_one({"user": username})
    
    # Get the list of movie titles from the user's watchlist
    movie_titles = user_watchlist.get("movies", []) if user_watchlist else []

    # Retrieve the movie details from the register_movie collection using the movie titles
    movies = Movie.objects.filter(title__in=movie_titles)

    # Calculate the average rating for each movie
    ratings_collection = db["ratings"]
    for movie in movies:
        ratings = list(ratings_collection.find({"movie": movie.title}))
        all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats

        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
        movie.avg_rating = avg_rating

    context = {
        'movies': movies,
        'logged_in_user_profile_pic_url': logged_in_user_profile_pic_url,

    }

    return render(request, 'favlist.html', context)






from django.shortcuts import render
from .models import Movie

def index(request):
    # Get all unique genres from the database
    genres = Movie.objects.values_list('genre', flat=True).distinct()

    context = {'genres': genres}
    return render(request, 'index.html', context)



from django.shortcuts import render
from .models import Movie

def movies_by_genre(request, genre):
    username = request.session.get('username')

    # Retrieve the user's watchlist from MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    watchlist_collection = db["watchlist"]
    user_watchlist = watchlist_collection.find_one({"user": username})
    
    # Get the list of movie titles from the user's watchlist
    movie_titles = user_watchlist.get("movies", []) if user_watchlist else []

    # Retrieve the movie details from the register_movie collection using the movie titles
    movies = Movie.objects.filter(title__in=movie_titles)

    # Calculate the average rating for each movie
    ratings_collection = db["ratings"]
    for movie in movies:
        ratings = list(ratings_collection.find({"movie": movie.title}))
        all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats

        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
        movie.avg_rating = avg_rating
    movies = Movie.objects.filter(genre__icontains=genre)
    context = {'movies': movies, 'selected_genre': genre, 'cine_rating' : avg}
    return render(request, 'movies_by_genre.html', context)






from django.shortcuts import render
from pymongo import MongoClient
from django.conf import settings
from .models import Movie
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend(request, imdb_id):
    # Retrieve the username from the session



    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]
    movie = get_object_or_404(Movie, imdbID=imdb_id)
    selected_movie = get_object_or_404(Movie, imdbID=imdb_id)

    # Get the genre of the selected movie
    selected_genre = selected_movie.genre

    # Retrieve movies with the same genre but exclude the selected movie
    recommended_movies = Movie.objects.filter(genre=selected_genre).exclude(imdbID=imdb_id)
    
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.cleaned_data['text']
            username = request.session.get('username')
            username1 = request.user.username if request.user.is_authenticated else None

            # Check if the user has already submitted a review
            reviews_collection = db["reviews"]
            existing_review = reviews_collection.find_one({"user": username, "movie": movie.title})

            if existing_review:
                # User has already submitted a review
                response_data = {
                    'message': "You have already submitted a review for this movie."
                }
                return JsonResponse(response_data, status=400)  # Return a bad request response with the message
            review_id = str(ObjectId())

            # Save the review to the database

            review_data = {
                'user': username,
                'movie': movie.title,
                '_id': review_id,  # Store the MongoDB ObjectId in the _id field
                'id': review_id,
                'text': review,
                'imdbID': imdb_id,
            }
            reviews_collection.insert_one(review_data)

            # Construct the response data
            response_data = {
                'my_review': review,

            }

            # Redirect back to movie details page
            return JsonResponse(response_data)

    elif request.method == 'PUT':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        existing_review = reviews_collection.find_one({'id': ObjectId(review_id)})

        if existing_review:
            data = json.loads(request.body.decode('utf-8'))
            updated_review_text = data.get('text', '')

            # Update the review in the MongoDB collection using update_one
            result = reviews_collection.update_one(
                {'id': ObjectId(review_id)},
                {'$set': {'text': updated_review_text}}
            )

            if result.modified_count > 0:
                # Return a success response if the document was updated successfully
                return JsonResponse({'message': 'Review updated successfully'})
            else:
                # Return an error response if the document was not updated
                return JsonResponse({'error': 'Failed to update review'}, status=400)
        else:
            return JsonResponse({'message': 'Review not found'}, status=404)

    elif request.method == 'DELETE':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        result = reviews_collection.delete_one({'id': ObjectId(review_id)})

        if result.deleted_count > 0:
            # Return a success response if the document was deleted successfully
            return HttpResponse(status=204)  # Return a success response with no content
        else:
            # Return an error response if the document was not found or not deleted
            return JsonResponse({'error': 'Review not found'}, status=404)
    username = request.session.get('username')
        # Query the existing reviews for the movie
    reviews_collection = db["reviews"]
    reviews = list(reviews_collection.find({"movie": movie.title}))

    # Query the existing ratings
    ratings_collection = db["ratings"]
    ratings = list(ratings_collection.find({"user": username, "movie": movie.title}))

    # Calculate the average rating for the movie
    all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
    avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None

    # Retrieve the username from the session
    username = request.session.get('username')

    # Find the user's own rating
    my_rating = None
    for rating in ratings:
        if rating['user'] == username:
            my_rating = rating['rate']
            break

    my_review = None
    for review in reviews:
        if review['user'] == username:
            my_review = review['text']
            break











    username = request.session.get('username')

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    ratings_collection = db["ratings"]

    # Retrieve all ratings data from MongoDB
    ratings_data = list(ratings_collection.find())

    # Create a DataFrame from the ratings data
    ratings_df = pd.DataFrame(ratings_data)

    print("Ratings Data:")
    print(ratings_df)

    if not ratings_df.empty:
        # Create a user-item matrix with users as rows and movies as columns
        user_item_matrix = ratings_df.pivot(index='user', columns='movie', values='rate').fillna(0)

        print("User-Item Matrix:")
        print(user_item_matrix)

        # Calculate user similarity based on cosine similarity
        user_similarity_matrix = cosine_similarity(user_item_matrix)

        print("User Similarity Matrix:")
        print(user_similarity_matrix)
        print(username)

        # Get the ratings of the logged-in user
        if username in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[username]
            print(user_ratings)
# You can set it to None or handle it as needed
        
# Find similar users by sorting the user similarity matrix
            similar_users_indices10 = np.argsort(-user_similarity_matrix[user_item_matrix.index == username])
            print(similar_users_indices10)
            num_users = user_item_matrix.shape[0]  # Total number of users
            percentage_to_select = 90  # Change this to your desired percentage
            N = int(num_users * (percentage_to_select / 100))


# Extract the indices of similar users excluding the current user
            similar_user_indices1 = similar_users_indices10[0]
            similar_user_indices = similar_user_indices1[:N]


            similar_user_indices = similar_user_indices[similar_user_indices != user_item_matrix.index.get_loc(username)]
            print(similar_user_indices)
            user_names = user_item_matrix.index.to_numpy()

            user_at_index_1 = user_names[similar_user_indices[0]]
            user_at_index_2 = user_names[similar_user_indices[1]]


            print("User at index 1:", user_at_index_1)
            print("User at index 2:", user_at_index_2)



# Get the names of similar users
            user_names = user_item_matrix.index.to_numpy()
            similar_user_names = user_names[similar_user_indices]
            print(similar_user_names )


# Now similar_user_names contains the names of users similar to the logged-in user, including the current user

# Filter the user-item matrix to include only similar users
# Create a boolean mask to filter the user-item matrix
            similar_users_mask = user_item_matrix.index.isin(similar_user_names)

# Apply the mask to get the user-item matrix for similar users
            similar_user_item_matrix = user_item_matrix[similar_users_mask]
            print("similar_user_item_matrix" )

            print(similar_user_item_matrix )
            similar_user_item_matrix1 = similar_user_item_matrix.astype(float)


# Calculate the mean ratings for each movie
            recommended_movie_ratings = similar_user_item_matrix1.mean()
            recommended_movie_ratings = recommended_movie_ratings.apply(str)

            print(recommended_movie_ratings )

# Filter out movies the user has already rated

# Sort recommended movies by rating in descending order
            recommended_movie_ratings = recommended_movie_ratings.sort_values(ascending=False)
            selected_movie = get_object_or_404(Movie, imdbID=imdb_id)
            print("selected_movie:", selected_movie)
            print("Top Recommended Movie Ratings:")
            print(recommended_movie_ratings)

# Get the top N recommended movies
            top_n_recommendations = recommended_movie_ratings.head(10)

            print("Top N Recommended Movies:")
            print(top_n_recommendations)

# Fetch movie objects for recommended movie titles
# Create a dictionary mapping movie titles to their ratings
# Create a dictionary mapping movie titles to their ratings
            movie_ratings_dict = dict(zip(recommended_movie_ratings.index, recommended_movie_ratings))

# Print the movie_ratings_dict to check its content
            print(movie_ratings_dict)

# Fetch movie objects for recommended movie titles
# Fetch movie objects for recommended movie titles
            recommended_movie_objects1 = Movie.objects.filter(title__in=top_n_recommendations.index)

# Print recommended_movie_objects before sorting
            print("Recommended Movies before sorting:")
            print(recommended_movie_objects1)

# Sort the recommended_movie_objects based on ratings (descending order)
# Sort the recommended_movie_objects based on ratings (descending order) and exclude the selected movie
            recommended_movie_objects12 = sorted(
            [movie for movie in recommended_movie_objects1 if movie != selected_movie], 
            key=lambda movie: movie_ratings_dict.get(movie.title, 0),
            reverse=True
            )
            for movie in recommended_movie_objects12:
                title_parts = movie.title.split(" : ")
                if len(title_parts) > 1:
                    movie.display_title = "<br>".join(title_parts)
                else:
                    movie.display_title = movie.title

  
            recommended_movie_objects = sorted(movie_ratings_dict.items(), key=lambda x: x[1], reverse=True)
        

# Print recommended_movie_objects after sorting
            print("Recommended Movies after sorting:")
            print(recommended_movie_objects)
        else:
            user_ratings = None  # You can set it to None or handle it as needed
        
            # Fetch average ratings for movies from the database
            collection = db["register_movie"]
            ratings_collection = db["ratings"]

            movies = Movie.objects.all()
            movie_ratings_dict = {}

            for movie in movies:
                ratings = list(ratings_collection.find({"movie": movie.title}))
                all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats

                avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
                movie_ratings_dict[movie.title] = avg_rating

            # Sort the movie_ratings_dict to get recommendations
            top_n_recommendations = dict(sorted(movie_ratings_dict.items(), key=lambda item: item[1], reverse=True)[:10])

            print(top_n_recommendations)

            # Fetch movie objects for recommended movie titles
            recommended_movie_objects = Movie.objects.filter(title__in=top_n_recommendations.keys())

            print(recommended_movie_objects)



    else:
        # Handle the case where there are no ratings
        recommended_movie_objects = []

    context = {
        'movie': movie,
        'reviews': reviews,  # Add reviews to the context
        'my_review': my_review,
        'ratings': ratings,
        'avg_rating': avg_rating,
        'my_rating': my_rating,
        'recommended_movies': recommended_movie_objects12,
        'top_n_recommendations': top_n_recommendations,
        'movie_ratings_dict': movie_ratings_dict,
        
    }

    return render(request, 'movie_details.html', context)
    


def login_success(request):
    return render (request, 'login_success.html')




from django.shortcuts import render
from pymongo import MongoClient
from django.conf import settings
from .models import Movie
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .models import Movie  # Import your Movie model

def itemtoitem(request):


        # Connect to MongoDB and retrieve movie data
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["register_movie"]
    movie = get_object_or_404(Movie, imdbID=imdb_id)
    selected_movie = get_object_or_404(Movie, imdbID=imdb_id)

    # Get the genre of the selected movie
    selected_genre = selected_movie.genre

    # Retrieve movies with the same genre but exclude the selected movie
    recommended_movies = Movie.objects.filter(genre=selected_genre).exclude(imdbID=imdb_id)
    
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.cleaned_data['text']
            username = request.session.get('username')
            username1 = request.user.username if request.user.is_authenticated else None

            # Check if the user has already submitted a review
            reviews_collection = db["reviews"]
            existing_review = reviews_collection.find_one({"user": username, "movie": movie.title})

            if existing_review:
                # User has already submitted a review
                response_data = {
                    'message': "You have already submitted a review for this movie."
                }
                return JsonResponse(response_data, status=400)  # Return a bad request response with the message
            review_id = str(ObjectId())

            # Save the review to the database

            review_data = {
                'user': username,
                'movie': movie.title,
                '_id': review_id,  # Store the MongoDB ObjectId in the _id field
                'id': review_id,
                'text': review,
                'imdbID': imdb_id,
            }
            reviews_collection.insert_one(review_data)

            # Construct the response data
            response_data = {
                'my_review': review,

            }

            # Redirect back to movie details page
            return JsonResponse(response_data)

    elif request.method == 'PUT':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        existing_review = reviews_collection.find_one({'id': ObjectId(review_id)})

        if existing_review:
            data = json.loads(request.body.decode('utf-8'))
            updated_review_text = data.get('text', '')

            # Update the review in the MongoDB collection using update_one
            result = reviews_collection.update_one(
                {'id': ObjectId(review_id)},
                {'$set': {'text': updated_review_text}}
            )

            if result.modified_count > 0:
                # Return a success response if the document was updated successfully
                return JsonResponse({'message': 'Review updated successfully'})
            else:
                # Return an error response if the document was not updated
                return JsonResponse({'error': 'Failed to update review'}, status=400)
        else:
            return JsonResponse({'message': 'Review not found'}, status=404)

    elif request.method == 'DELETE':
        review_id = request.POST.get('review_id')  # Assuming the review_id is sent in the request
        reviews_collection = db["reviews"]
        result = reviews_collection.delete_one({'id': ObjectId(review_id)})

        if result.deleted_count > 0:
            # Return a success response if the document was deleted successfully
            return HttpResponse(status=204)  # Return a success response with no content
        else:
            # Return an error response if the document was not found or not deleted
            return JsonResponse({'error': 'Review not found'}, status=404)
    username = request.session.get('username')

    # Query the existing reviews for the movie
    reviews_collection = db["reviews"]
    reviews = list(reviews_collection.find({"movie": movie.title}))

    # Query the existing ratings
    ratings_collection = db["ratings"]
    ratings = list(ratings_collection.find({"user": username, "movie": movie.title}))

    # Calculate the average rating for the movie
    all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
    avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None

    # Retrieve the username from the session
    username = request.session.get('username')

    # Find the user's own rating
    my_rating = None
    for rating in ratings:
        if rating['user'] == username:
            my_rating = rating['rate']
            break

    my_review = None
    for review in reviews:
        if review['user'] == username:
            my_review = review['text']
            break
    












    # Retrieve the username from the session
    username = request.session.get('username')

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    ratings_collection = db["ratings"]

    # Retrieve all ratings data from MongoDB
    ratings_data = list(ratings_collection.find())

    # Create a DataFrame from the ratings data
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df['rate'] = pd.to_numeric(ratings_df['rate'])
    print(ratings_df['rate'])

    if not ratings_df.empty:
        # Create a user-item matrix with users as rows and movies as columns
        user_item_matrix = ratings_df.pivot(index='user', columns='movie', values='rate').fillna(0)
        print("user_item_matrix")

        print(user_item_matrix)
        # Calculate item similarity based on cosine similarity
        item_similarity_matrix = cosine_similarity(user_item_matrix.T)
        print("item_similarity_matrix")
        print(item_similarity_matrix)
       

        # Convert the similarity matrix into a DataFrame
        item_similarity_df = pd.DataFrame(item_similarity_matrix, columns=user_item_matrix.columns, index=user_item_matrix.columns)
        print("item_similarity_df")
        print(item_similarity_df)

        # Get the user's rated movies
        user_rated_movies = user_item_matrix.loc[username].index[user_item_matrix.loc[username] > 0]
        print("user rated movies")
        print(user_rated_movies)

        # Get similar movies for each user-rated movie
        recommended_movies = []
        for movie in user_rated_movies:
            similar_movies = item_similarity_df[movie].sort_values(ascending=False)
            recommended_movies.extend(similar_movies.index[1:])  # Exclude the current movie
        print("recommended movies before removing duplicates and movies the user has already rated")    
        print(recommended_movies)
        # Remove duplicates and movies the user has already rated
        recommended_movies = [movie for movie in recommended_movies if movie not in user_rated_movies]
        print("recommended movies after  removing duplicates and movies the user has already rated")    
        print(recommended_movies)
        # Get movie objects for recommended movie titles
        recommended_movie_objects = Movie.objects.filter(title__in=recommended_movies)

    else:
        # Handle the case where there are no ratings
        recommended_movie_objects = []

    context = {
        'movie': movie,
        'reviews': reviews,  # Add reviews to the context
        'my_review': my_review,
        'ratings': ratings,
        'avg_rating': avg_rating,
        'my_rating': my_rating,
        'selected_movie': selected_movie,
        'recommended_movies': recommended_movies,
    }

    return render(request, 'movie_details.html', context)



# views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm

def custom_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('admin_dashboard')  # Redirect to your admin dashboard view
            else:
                # Invalid credentials
                form.add_error(None, 'Invalid username or password')
    else:
        form = LoginForm()
    return render(request, 'custom_login.html', {'form': form})



from django.shortcuts import render
from .models import Movie
from pymongo import MongoClient
import statistics

def top_10_movies(request):
    # Connect to your MongoDB database
    client = MongoClient("mongodb://localhost:27017")
    db = client["regis"]
    collection = db["regis1"]
    logged_in_user = request.session.get('username')
    print(logged_in_user)

    logged_in_user_data = collection.find_one({"username": logged_in_user})

    logged_in_user_profile_pic = logged_in_user_data.get("profile_pic") if logged_in_user_data else None

    if logged_in_user_profile_pic:
        logged_in_user_profile_pic_url = f"{settings.MEDIA_URL}{logged_in_user_profile_pic}".replace('\\', '/')
    else:
        logged_in_user_profile_pic_url = None
    client = MongoClient("mongodb://localhost:27017")
    db = client['regis']
    collection = db['register_movie']
    ratings_collection = db['ratings']

    # Retrieve all movies
    movies = Movie.objects.all()

    # Calculate the average rating for each movie
    for movie in movies:
        ratings = list(ratings_collection.find({"movie": movie.title}))
        all_ratings = [float(review['rate']) for review in ratings]  # Convert strings to floats
        avg_rating = round(statistics.mean(all_ratings), 1) if all_ratings else None
        movie.avg_rating = avg_rating

    # Find the top 10 rated movies based on avg_rating
    top_rated_movies = sorted(
    movies,
    key=lambda x: x.avg_rating if x.avg_rating is not None else float('-inf'),
    reverse=True
    )[:10]

    context = {'top_rated_movies': top_rated_movies,'logged_in_user_profile_pic_url': logged_in_user_profile_pic_url,}
    return render(request, 'top10.html', context)

