from django.shortcuts import render
import mysql.connector as sql
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from django.http import HttpResponseRedirect
cod=''
user=''

# Create your views here.
def verific(request):
    global cod,user
    user = request.GET.get('name', '')
    if request.method == "POST":
        m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
        cursor = m.cursor()

        d = request.POST
        for key, value in d.items():
            if key == "codigo":
                cod = value
            if key == "user":
                user = value
        # AES ECB mode without IV


        key = 'AAAAAAAAAAAAAAAA'  # Must Be 16 char for AES128

        def encrypt(raw):
            raw = pad(raw.encode(), 16)
            cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
            return base64.b64encode(cipher.encrypt(raw))


        encrypted = encrypt(user)
        user = encrypted.decode("utf-8", "ignore")
        c = "select codigo from utilizador where username='{}' and codigo='{}'".format(user, cod)
        cursor.execute(c)
        for row in c:
            cod= row[0]
        print(cod)
        t = tuple(cursor.fetchall())

        if t==():
            def decrypt(enc):
                enc = base64.b64decode(enc)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return unpad(cipher.decrypt(enc), 16)

            user = decrypt(user)
            user = user.decode("utf-8", "ignore")
            return render(request,'verificar.html',{'posts': user})
        else:

            def decrypt(enc):
                enc = base64.b64decode(enc)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return unpad(cipher.decrypt(enc), 16)

            user = decrypt(user)
            user = user.decode("utf-8", "ignore")
            return HttpResponseRedirect("/home_principal/?name={}".format(user))

    return render(request,'verificar.html',{'posts': user})
