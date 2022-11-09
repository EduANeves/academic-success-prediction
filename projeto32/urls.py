"""website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from signup.views import signaction
from login.views import loginaction
from verificar.views import verific
from home_principal.views import home_prin
from prev_resultado.views import prev_result
from recu_conta.views import recu_cont
from new_pass.views import newpass
from apagar.views import apaga
urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/',signaction),
    path('login/',loginaction),
    path('verificar/', verific),
    path('home_principal/', home_prin),
    path('prev_resultado/', prev_result),
    path('recu_conta/', recu_cont),
    path('new_pass/', newpass),
    path('apagar/', apaga),
]
