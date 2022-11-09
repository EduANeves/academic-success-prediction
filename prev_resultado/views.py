from django.shortcuts import render
import numpy as np
import seaborn as sns
import pandas as pd
import mysql.connector as sql
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from django.http import HttpResponseRedirect
user = ''
svm = ''
rede =''
radio=''
id=''
nome =''
data=''
# Create your views here.
def prev_result(request):
    global user, svm, rede, radio,id,nome,data
    user = request.GET.get('name', '')
    rede = request.GET.get('nota', '')
    svm = request.GET.get('svm', '')

    if request.method == "POST":
        m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
        cursor = m.cursor()
        d = request.POST
        for key, value in d.items():
            if key == "nota":
                rede = value
            if key == "svm":
                svm = value
            if key == "guardar":
                radio = value
        if radio == 'sim':
            for key, value in d.items():
                if key == "id":
                    id = value
                if key == "nome":
                    nome = value
            key = 'AAAAAAAAAAAAAAAA'
            def encrypt(raw):
                raw = pad(raw.encode(), 16)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return base64.b64encode(cipher.encrypt(raw))

            encrypted = encrypt(user)
            user = encrypted.decode("utf-8", "ignore")
            encrypted = encrypt(id)
            id = encrypted.decode("utf-8", "ignore")
            encrypted = encrypt(nome)
            nome = encrypted.decode("utf-8", "ignore")
            encrypted = encrypt(svm)
            svm = encrypted.decode("utf-8", "ignore")
            from datetime import datetime

            data = datetime.today().strftime('%Y-%m-%d')
            c = "insert into historico(username,id_aluno,svm,avaliacao,nome_aluno,data) Values('{}','{}','{}','{}','{}','{}')".format(user, id, svm, rede, nome, data)
            cursor.execute(c)
            m.commit()
            t = tuple(cursor.fetchall())
            if t == ():
                def decrypt(enc):
                    enc = base64.b64decode(enc)
                    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                    return unpad(cipher.decrypt(enc), 16)

                email = decrypt(user)
                user = email.decode("utf-8", "ignore")
                return HttpResponseRedirect("/home_principal/?name={}".format(user))
        return HttpResponseRedirect("/home_principal/?name={}".format(user))
    return render(request, 'prev_resultado.html', {'user': user, 'posts': rede, 'svm': svm})