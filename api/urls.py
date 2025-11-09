# -*- coding: utf-8 -*-
from django.urls import path
from .views import play_view, stop_view

urlpatterns = [
    path("api/play", play_view, name="api_play"),
    path("api/stop", stop_view, name="api_stop"),
]
