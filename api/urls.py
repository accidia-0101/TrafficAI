# events/urls.py
# -*- coding: utf-8 -*-
from django.urls import path

from api.views import play_view, stop_view,alerts_stream


urlpatterns = [
    path("api/play", play_view),
    path("api/stop", stop_view),
    path("sse/alerts", alerts_stream),

]
