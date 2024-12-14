
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns=[
    #basic 
    path('',views.index,name='index'), # no ai recomendations
    path('products',views.generate_recommendations,name='prducts'), # with ai recomendations 


    #authentecation urls
    path('register/', views.register, name='register'),
    path('login', views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),


    # Place holder
    path('about',views.index,name='about'),
    path('article',views.index,name='article'),
    path('faq',views.index,name='faq'),
    path('contact',views.index,name='contact'),
    path('news',views.index,name='news'),
    path('blog',views.index,name='blog'),
    path('delivery',views.index,name='delivery'),
]
