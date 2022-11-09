from django.shortcuts import render
import mysql.connector as sql
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from django.http import HttpResponseRedirect
cod=''
pass1=''
pass2=''
aviso=''
# Create your views here.
def newpass(request):
    global cod,pass1,pass2
    user = request.GET.get('name', '')
    aviso = ''
    if request.method == "POST":
        m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
        cursor = m.cursor()

        d = request.POST
        for key, value in d.items():
            if key == "cod":
                cod = value
            if key == "password":
                pass1 = value
            if key == "password2":
                pass2 = value
        # AES ECB mode without IV
        if len(pass1) < 9:
            aviso = 'Password contem menos de 9 caracteres!!'
            return render(request, 'newpass_page.html', {'posts': user, 'aviso': aviso})
        if pass1 != pass2:
            aviso = 'Password não são iguais!!'
            return render(request, 'newpass_page.html', {'posts': user, 'aviso': aviso})
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
            aviso = 'Código errado!!'

            def decrypt(enc):
                enc = base64.b64decode(enc)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return unpad(cipher.decrypt(enc), 16)

            user = decrypt(user)
            user = user.decode("utf-8", "ignore")
            return render(request,'newpass_page.html',{'posts': user,'aviso': aviso})
        else:
            pass1 = encrypt(pass1)
            pass1 = pass1.decode("utf-8", "ignore")
            c = "update utilizador SET password='{}' where username='{}'".format(pass1, user)
            cursor.execute(c)
            m.commit()
            def decrypt(enc):
                enc = base64.b64decode(enc)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return unpad(cipher.decrypt(enc), 16)

            user = decrypt(user)
            user = user.decode("utf-8", "ignore")
            return HttpResponseRedirect("/login/")

    return render(request,'newpass_page.html',{'posts': user,'aviso': aviso})
