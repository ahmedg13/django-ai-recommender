from django.shortcuts import render ,redirect

#user login modules
from django.contrib.auth import authenticate, login ,logout 
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

#models
from products.models import Product ,UserInteraction,ProductCategory ,ProductBrand

#recommend engine classes & modules
import pandas as pd
from products.recommendtion_engine import MLRecommendationEngine

import datetime


# product page with no recommendation filtering
def index(request):
    
    if  request.user.id == None  :
        return redirect("login")
        
    # if request.method == "POST":
    #     product = Product.objects.get( id =request.POST["product"]  )
    #     user_int = UserInteraction(user=request.user ,product=product , interaction_type=request.POST["interaction"]  ,timestamp= datetime.datetime.now() ,  ).save()
            


    return redirect("prducts")

# product page with recommendation filtering
def generate_recommendations(request):
    """
    Steps:
    1. Fetch user interactions (from UserInteractions model)
    2. Train recommendation model
    3. Generate personalized recommendations
    """

    if request.method == "POST":
        product = Product.objects.get( id =request.POST["product"]  )
        user_intraction = UserInteraction(user=request.user ,product=product , interaction_type=request.POST["interaction"]  ,timestamp= datetime.datetime.now() ,  ).save()
        

    if  request.user.id == None  :
        return redirect("login")

    # Fetch interaction data from database
    interactions_df = pd.DataFrame(list(
        UserInteraction.objects.values('user_id', 'product_id', 'weight') ) )

    # Initialize and train recommendation engiaboutne
    recommendation_engine = MLRecommendationEngine()

    recommendation_engine.train_recommendation_model(interactions_df)
    
    # Generate recommendations for current user
    recommendations = recommendation_engine.get_recommendations(request.user.id ,99)
    
    # Fetch recommended product objects
    recommended_products = Product.objects.filter(id__in=recommendations , )


    
    return render(request, 'pages/products.html', {
        'title':'Products',
        'products': recommended_products,
        "categorys" :ProductCategory.objects.all() ,
        "brands" : ProductBrand.objects.all()
    })



def user_login(request):

    # Check if the user is aleardy logined if so redirects him to the index page
    if  request.user.id != None :
        return redirect('index')

    # if the login is valid logs the user and sends him to the index page
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            # Return an 'invalid login' error message
            return render(request, 'pages/auth/login.html', {'error': 'Invalid credentials'})
    
    return render(request, 'pages/auth/login.html' , {"title" :"login"})


def register(request):

    # Check if the user is aleardy logined if so redirects him to the index page
    if  request.user.id != None  :
        return redirect('index')

    # Use django functions to register him becouse it's easier that way
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        
        if form.is_valid():
            user = form.save()
            
            login(request, user)
            return redirect('index')  # Redirect to index page after registration
    else:
        form = UserCreationForm()
    return render(request, 'pages/auth/register.html', {'form': form ,'title':"register"})


# logs the user out (duh it's self explanatory)
def user_logout(request):
    logout(request)
    return redirect('login')