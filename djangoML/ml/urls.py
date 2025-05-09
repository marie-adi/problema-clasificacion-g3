from django.urls import path
from . import views

app_name = 'ml'
urlpatterns = [
    path('', views.form, name='ml'),
]