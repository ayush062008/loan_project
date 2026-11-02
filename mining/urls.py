from django.urls import path

from . import views

urlpatterns = [
    path("", views.upload, name="upload"),
    path("sample/", views.use_sample, name="use_sample"),
    path("dataset/<str:run_id>/", views.dataset, name="dataset"),
    path("results/<str:run_id>/", views.summary, name="summary"),
    path("results/<str:run_id>/dashboard/", views.dashboard, name="dashboard"),
    path("results/<str:run_id>/predict/", views.predict, name="predict"),
    path("results/<str:run_id>/predict/api/", views.predict_api, name="predict_api"),
]
