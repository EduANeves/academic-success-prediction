from django.shortcuts import render,redirect
import mysql.connector as sql
from django.http import HttpResponseRedirect
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
em=''
pwd=''

nome=''
aviso=''
# Create your views here.
def recu_cont(request):
    global em,pwd,nome,aviso
    aviso=''
    if request.method=="POST":
        m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
        cursor=m.cursor()
        d=request.POST
        for key,value in d.items():
            if key=="email":
                em=value


            # AES ECB mode without IV

        data = 'Olá eu sou o joão'
        key = 'AAAAAAAAAAAAAAAA'  # Must Be 16 char for AES128

        def encrypt(raw):
            raw = pad(raw.encode(), 16)
            cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
            return base64.b64encode(cipher.encrypt(raw))



        encrypted = encrypt(em)
        em = encrypted.decode("utf-8", "ignore")


        c="select nome from utilizador where email='{}' ".format(em)
        cursor.execute(c)


        t=tuple(cursor.fetchall())
        for row in t:
            nome= row[0]

        c="select username from utilizador where email='{}' ".format(em)
        cursor.execute(c)


        t=tuple(cursor.fetchall())
        for row in t:
            pwd= row[0]

        import string
        import random

        S = 6
        ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))

        c = "update utilizador SET codigo='{}' where username='{}'".format(ran, pwd)
        cursor.execute(c)
        m.commit()

        if t==():
            aviso='Email incorreto!'
            return render(request,'recu_page.html',{'aviso': aviso})
        else:
            def decrypt(enc):
                enc = base64.b64decode(enc)
                cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
                return unpad(cipher.decrypt(enc), 16)

            email = decrypt(em)
            email = email.decode("utf-8", "ignore")

            f = "select nome from utilizador where email='{}'".format(em)
            cursor.execute(f)

            g = cursor.fetchall()
            for row in g:
                nome = row[0]

            nome = decrypt(nome)
            nome = nome.decode("utf-8", "ignore")
            em = decrypt(em)
            em = em.decode("utf-8", "ignore")

            # me == my email address
            # you == recipient's email address
            me = "vidtekejr@gmail.com"
            you = email

            # Create message container - the correct MIME type is multipart/alternative.
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "Email de Confirmação | PSAE"
            msg['From'] = me
            msg['To'] = you



            # Create the body of the message (a plain-text and an HTML version).
            text = ""
            html = """\
            <html lang="en">
            <head>
                <meta charset='UTF-8'>
                <style>
                    h2{
                        font-size: 30px;
                        color: black;
                        font-family: 'Roboto',sans-serif;
                        text-align: center;}
                    h3{
                        font-size: 15px;
                        color: white;
                        font-family: 'Roboto',sans-serif;}
                    h4{
                        font-size: 16px;
                        color: black;
                        font-family: Arial;
                        text-align: center;}
                    .p{
                        font-family: 'Roboto',sans-serif;
                        font-size: 13px;
                        color: grey;}
                    .hr{border: 0; border-top: 1px solid #e2e2e2;}
                    .codigo{
                         border: 2px solid #e2e2e2;
                         width: 25%;
                         border-radius: 8px;
                         vertical-align:middle;
                         height: 100%;
                         text-align: center;
                         padding: 10px 40px;}
                    .sizecod{
                         width: 100%;
                         height: 25%;
                         font-size: 30px;
                         font-family: 'Roboto',sans-serif;
                         color: black;}
                    .divprincipal{
                        width: 100%;
                        height: 100%;
                        background-color: #e2e2e2;}
                    .divmeio{
                        width: 60%;
                        vertical-align:middle;
                        position: relative;
                        height: 60%;
                        background-color: white;}
                    .divimg{
                        width: 100%;
                        position: relative;
                        height: 100%;
                        background-color: #94C5FF;}
                     .divbaixo{
                        width: 60%;
                        vertical-align: middle;
                        position: relative;
                        height: 100%;
                        background-color: black;}
                </style>
            </head>
            <body>
                <div class= 'divprincipal'>
                    <center>
                        <div class='divmeio'>
                            <div class='divimg'>
                                <img src='https://paol.iscap.ipp.pt/recles/images/imagens/logo_ulht_horizontal_v2-01.png' width=100% height=50%></div>
                                      <h2>Bem vindo de novo,""" + nome + """!</h2>
                                      <h4>Utiliza o código para recuperar a conta.</h4><br>
                                      <div class ='sizecod'><div class ='codigo'>""" + ran + """
                                      </div>
                                      </div><br><br>
                            <hr class='hr'><br>
                            <p class='p'>Recebeu este email porque tentou fazer login na sua conta de PSAE. Se não pediu para entrar na sua conta, pode ignorar este email.</p></div>
                        <div class='divbaixo'>

                        </div>
                        <h3>O que vamos colocar aqui.</h3>
                    </center>
                </div>
            </body>
            </html>
            """

            # Record the MIME types of both parts - text/plain and text/html.
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')

            # Attach parts into message container.
            # According to RFC 2046, the last part of a multipart message, in this case
            # the HTML message, is best and preferred.
            msg.attach(part1)
            msg.attach(part2)
            # Send the message via local SMTP server.
            mail = smtplib.SMTP('smtp.gmail.com', 587)

            mail.ehlo()

            mail.starttls()

            mail.login('vidtekejr@gmail.com', 'ejrvidtek')
            mail.sendmail(me, you, msg.as_string())
            mail.quit()
            print('Enviado')



            pwd = decrypt(pwd)
            pwd = pwd.decode("utf-8", "ignore")

            return HttpResponseRedirect("/new_pass/?name={}".format(pwd))

    return render(request,'recu_page.html',{'aviso': aviso})
