from django.shortcuts import render
from django.http import HttpResponseRedirect
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
import mysql.connector as sql
import numpy as np
import seaborn as sns
import pandas as pd
user=''
id=0
gg=''
hh=0
# Create your views here.
def apaga(request):
    global user,id,gg,hh
    user = request.GET.get('user', )
    id = request.GET.get("id", 0)
    hh = int(id)
    print (id)
    m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
    cursor2 = m.cursor()
    user = user.strip("'")
    key = 'AAAAAAAAAAAAAAAA'
    def encrypt(raw):
        raw = pad(raw.encode(), 16)
        cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
        return base64.b64encode(cipher.encrypt(raw))

    encrypted = encrypt(user)
    gg = encrypted.decode("utf-8", "ignore")

    c = "DELETE FROM historico WHERE username='{}' and id_historico={};".format(gg, id)
    print(c)
    cursor2.execute(c)
    m.commit()


    return HttpResponseRedirect("/home_principal/?name={}".format(user))
