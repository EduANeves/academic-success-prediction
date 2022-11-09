from django.shortcuts import render
import mysql.connector as sql
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
fn=''
ln=''
user=''
em=''
pwd=''
cod='AAAZZZ'
email=''
aviso=''
pwd1=''
# Create your views here.
def signaction(request):
    global fn,ln,user,em,pwd,cod,email,aviso,pwd1
    aviso=''
    if request.method=="POST":
        m=sql.connect(host="localhost",user="root",passwd="EduN1305",database="projeto")
        cursor=m.cursor()
        d=request.POST
        for key,value in d.items():
            if key =="nome":
                fn=value
            if key == "sobrenome":
                ln=value
            if key == "username":
                user=value
            if key == "email":
                em=value
            if key =="password":
                pwd=value
            if key =="conpassword":
                pwd1=value

        if len(pwd) >= 9:
            if pwd==pwd1:

                # AES ECB mode without IV

                data = 'Olá eu sou o joão'
                key = 'AAAAAAAAAAAAAAAA'  # Must Be 16 char for AES128

                def encrypt(raw):
                    raw = pad(raw.encode(), 16)
                    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                    return base64.b64encode(cipher.encrypt(raw))


                encrypted = encrypt(user)
                user= encrypted.decode("utf-8", "ignore")
                encrypted = encrypt(pwd)
                pwd = encrypted.decode("utf-8", "ignore")
                encrypted = encrypt(em)
                em = encrypted.decode("utf-8", "ignore")
                encrypted = encrypt(fn)
                fn = encrypted.decode("utf-8", "ignore")
                encrypted = encrypt(ln)
                ln = encrypted.decode("utf-8", "ignore")
                c = "select email from utilizador where username='{}' and email='{}' ".format(user,em)
                cursor.execute(c)

                t = tuple(cursor.fetchall())
                for row in t:
                    email = row[0]
                print(email)
                if t == ():

                    c = "insert into utilizador Values('{}','{}','{}','{}','{}','{}')".format(user, pwd, em, fn, ln, cod)
                    cursor.execute(c)
                    m.commit()
                    t = tuple(cursor.fetchall())
                    if t == ():
                        aviso='Utilizador inserido!'
                        return render(request, 'signup_page.html',{'aviso': aviso})
                    else:
                        return render(request, 'login_page.html')

                else:
                    aviso='Utilizador já existente'
                    return render(request, 'signup_page.html',{'aviso': aviso})
            else:
                aviso='As Passwords são diferentes'
                return render(request, 'signup_page.html',{'aviso': aviso})
        else:
            aviso = 'Password tem menos de 9 caracteres'
            return render(request, 'signup_page.html', {'aviso': aviso})
    return render(request,'signup_page.html',{'aviso': aviso})

