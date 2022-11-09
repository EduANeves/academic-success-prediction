from django.http import HttpResponseRedirect
from django.shortcuts import render
import mysql.connector as sql
import csv
import pandas as pd
import base64
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from django.utils.html import format_html
caminho=''
user=''
funcao=''
escola=''
razao=''
rcs=''
nota_primeiro=''
genero=''
guardiao=''
rai=''
nota_segundo=''
idade=0
tempo=0
rrr=''
rh=''
est_semanal=0
rel_familiar = 0
rtf=''
reprovacao = 0
tempo_livre = 0
ref=''
rse=''
saidaamigos=0
educacaomae=0
rsf=''
consumodiario=0
educacaopai=0
ree=''
consumosemanal=0
ocupacaomae=''
rsx=''
saude=0
ocupacaopai=''
ri=''
faltas=0
svm=''
gg=''
id_historico=''
id_aluno=''
svm2=''
avaliacao2=''
nome_aluno=''
data=''
string2='<font color=red>oi</font>'
string=''
s=0
# Create your views here.
def home_prin(request):
    global s, string,string2,id_historico,id_aluno,svm2,avaliacao2,nome_aluno,data,gg, svm,caminho,user,funcao,escola,razao,rcs,nota_primeiro,genero,guardiao,rai,nota_segundo,idade,tempo,rrr,rh,est_semanal,rel_familiar,rtf,reprovacao,tempo_livre,ref, rse,saidaamigos,educacaomae,rsf,consumodiario,educacaopai,ree,consumosemanal,ocupacaomae,rsx,saude,ocupacaopai,ri,faltas
    user = request.GET.get('name', '')
    m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
    cursor2 = m.cursor()
    key = 'AAAAAAAAAAAAAAAA'
    string=''
    string='<table class="table3"> <tr class="tr3"><th class="th3">ID_Aluno</th><th class="th3">Nome</th><th class="th3">Descrição</th><th class="th3">Avaliação</th><th class="th3">Datas</th><th class="th3">Opções</th></tr>'

    format_html(string2)
    def encrypt(raw):
        raw = pad(raw.encode(), 16)
        cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
        return base64.b64encode(cipher.encrypt(raw))

    encrypted = encrypt(user)
    gg = encrypted.decode("utf-8", "ignore")

    def decrypt(enc):
        enc = base64.b64decode(enc)
        cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
        return unpad(cipher.decrypt(enc), 16)


    f = "select id_historico,id_aluno,svm,avaliacao,nome_aluno,data from historico where username='{}' ".format(gg)
    cursor2.execute(f)
    s=0
    g = cursor2.fetchall()
    for row in g:
        s+=1
        id_historico = row[0]
        id_aluno = row[1]
        svm2 = row[2]
        avaliacao2 = row[3]
        nome_aluno = row[4]
        data = row[5]
        id_aluno = decrypt(id_aluno)
        id_aluno = id_aluno.decode("utf-8", "ignore")
        svm2 = decrypt(svm2)
        svm2 = svm2.decode("utf-8", "ignore")
        nome_aluno = decrypt(nome_aluno)
        nome_aluno = nome_aluno.decode("utf-8", "ignore")

        if svm2=='Sucesso académico':
            if avaliacao2 <9.5:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=green>' + svm2 + '</font></td><td class="td3"><font color=red>' + str(avaliacao2) + '</font></td><td class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user="+ user + "&id="+ str(id_historico)  + ""'> Apagar</a></td></tr>'
            if avaliacao2 >=9.5 and  avaliacao2<=13.9:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=green>' + svm2 + '</font></td><td class="td3"><font color=orange>' + str(avaliacao2) + '</font></td><td class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user="+ user + "&id="+ str(id_historico)  + ""'> Apagar</a></td></tr>'
            if avaliacao2 >13.9:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=green>' + svm2 + '</font></td><td class="td3"><font color=green>' + str(avaliacao2) + '</font></td><td class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user="+ user + "&id="+ str(id_historico)  + ""'> Apagar</a></td></tr>'
        if svm2=='Insucesso académico':
            if avaliacao2 <9.5:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=red>' + svm2 + '</font></td><td  class="td3"><font color=red>' + str(avaliacao2) + '</font></td><td  class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user='"+ user + "'&id="+ str(id_historico) + ""'> Apagar</a></td></tr>'
            if avaliacao2 >=9.5 and  avaliacao2<=13.9:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=red>' + svm2 + '</font></td><td  class="td3"><font color=orange>' + str(avaliacao2) + '</font></td><td  class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user='"+ user + "'&id="+ str(id_historico) + ""'> Apagar</a></td></tr>'
            if avaliacao2 >13.9:
                string += '<tr class="tr3"><td class="td3">' + id_aluno + '</td><td class="td3">' + nome_aluno + '</td><td class="td3"><font color=red>' + svm2 + '</font></td><td  class="td3"><font color=green>' + str(avaliacao2) + '</font></td><td  class="td3">' + str(data) + '</td><td class="td3"><a class="h" href='"http://localhost:8000/apagar/?user='"+ user + "'&id="+ str(id_historico) + ""'> Apagar</a></td></tr>'


    string +='</table>'
    format_html(string)
    if s==0:
        string=''
    cursor2.close()
    if request.method == "POST":
        m = sql.connect(host="localhost", user="root", passwd="EduN1305", database="projeto")
        cursor = m.cursor()

        d = request.POST
        for key, value in d.items():
            if key == "funcao":
                funcao = value

        if(funcao=="prever"):
            for key, value in d.items():
                if key == "caminho":
                    caminho = value
                if key == "user":
                    user = value
                if key == "escola":
                    escola = value
                if key == "razao":
                    razao = value
                if key == "rcs":
                    rcs = value
                if key == "nota_primeiro":
                    nota_primeiro = value
                if key == "genero":
                    genero = value
                if key == "guardiao":
                    guardiao = value
                if key == "rai":
                    rai = value
                if key == "nota_segundo":
                    nota_segundo = value
                if key == "idade":
                    idade = value
                if key == "tempo":
                    tempo = value
                if key == "rrr":
                    rrr = value
                if key == "rh":
                    rh = value
                if key == "est_semanal":
                    est_semanal = value
                if key == "rel_familiar":
                    rel_familiar = value
                if key == "rtf":
                    rtf = value
                if key == "reprovacao":
                    reprovacao = value
                if key == "tempo_livre":
                    tempo_livre = value
                if key == "ref":
                    ref = value
                if key == "rse":
                    rse = value
                if key == "saidaamigos":
                    saidaamigos = value
                if key == "educacaomae":
                    educacaomae = value
                if key == "rsf":
                    rsf = value
                if key == "consumodiario":
                    consumodiario = value
                if key == "educacaopai":
                    educacaopai = value
                if key == "ree":
                    ree = value
                if key == "consumosemanal":
                    consumosemanal = value
                if key == "ocupacaomae":
                    ocupacaomae = value
                if key == "rsx":
                    rsx = value
                if key == "saude":
                    saude = value
                if key == "ocupacaopai":
                    ocupacaopai = value
                if key == "ri":
                    ri = value
                if key == "faltas":
                    faltas = value

            import csv

            url = 'C:/Users/eduar/PycharmProjects/Projeto3/Projeto3/template/prever.csv'

            with open(url, 'w', newline='') as student_file:
                writer = csv.writer(student_file)
                writer.writerow(
                    ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason",
                     "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities",
                     "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
                     "health", "absences", "G1", "G2", "G3"])
                writer.writerow(
                    [escola, genero, idade, rh, rtf, ref, educacaomae, educacaopai, ocupacaomae, ocupacaopai, razao, guardiao, tempo, est_semanal, reprovacao, rse, rsf,
                     ree, rsx, ri, rcs, rai, rrr, rel_familiar, tempo_livre, saidaamigos, consumodiario, consumosemanal, saude, faltas, nota_primeiro, nota_segundo, 0])
            caminho = 'C:/Users/eduar/PycharmProjects/Projeto3/Projeto3/template/' + caminho

            data = pd.read_csv(caminho)
            y_por = data.iloc[:, -1]

            data = data.drop(['G3'], axis=1)

            data.head()

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(data, y_por)
            print(X_train)
            print(X_test)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train['absences'] = scaler.fit_transform(X_train['absences'].values.reshape(-1, 1))
            X_test['absences'] = scaler.fit_transform(X_test['absences'].values.reshape(-1, 1))

            import tensorflow as tf

            school_vocab = ['MS', 'GP']
            school_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="school", vocabulary_list=school_vocab)

            sex_vocab = ['M', 'F']
            sex_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="sex", vocabulary_list=sex_vocab)

            age_vocab = [18, 16, 17, 20, 19, 15, 21, 22]
            age_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="age", vocabulary_list=age_vocab)

            address_vocab = ['R', 'U']
            address_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="address", vocabulary_list=address_vocab)

            famsize_vocab = ['GT3', 'LE3']
            famsize_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="famsize", vocabulary_list=famsize_vocab)

            Pstatus_vocab = ['T', 'A']
            Pstatus_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Pstatus", vocabulary_list=Pstatus_vocab)

            Medu_vocab = [3, 4, 1, 2, 0]
            Medu_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Medu", vocabulary_list=Medu_vocab)

            Fedu_vocab = [2, 1, 4, 3, 0]
            Fedu_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Fedu", vocabulary_list=Fedu_vocab)

            Mjob_vocab = ['services', 'other', 'teacher', 'at_home', 'health']
            Mjob_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Mjob", vocabulary_list=Mjob_vocab)

            Fjob_vocab = ['other', 'health', 'services', 'teacher', 'at_home']
            Fjob_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Fjob", vocabulary_list=Fjob_vocab)

            reason_vocab = ['course', 'reputation', 'other', 'home']
            reason_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="reason", vocabulary_list=reason_vocab)

            guardian_vocab = ['mother', 'father', 'other']
            guardian_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="guardian", vocabulary_list=guardian_vocab)

            traveltime_vocab = [1, 2, 4, 3]
            traveltime_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="traveltime", vocabulary_list=traveltime_vocab)

            studytime_vocab = [1, 4, 2, 3]
            studytime_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="studytime", vocabulary_list=studytime_vocab)

            failures_vocab = [1, 4, 2, 3]
            failures_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="failures", vocabulary_list=failures_vocab)

            schoolsup_vocab = ['no', 'yes']
            schoolsup_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="schoolsup", vocabulary_list=schoolsup_vocab)

            famsup_vocab = ['yes', 'no']
            famsup_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="famsup", vocabulary_list=famsup_vocab)

            paid_vocab = ['no', 'yes']
            paid_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="paid", vocabulary_list=paid_vocab)

            activities_vocab = ['no', 'yes']
            activities_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="activities", vocabulary_list=activities_vocab)

            nursery_vocab = ['yes', 'no']
            nursery_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="nursery", vocabulary_list=nursery_vocab)

            higher_vocab = ['yes', 'no']
            higher_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="higher", vocabulary_list=higher_vocab)

            internet_vocab = ['yes', 'no']
            internet_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="internet", vocabulary_list=internet_vocab)

            romantic_vocab = ['no', 'yes']
            romantic_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="romantic", vocabulary_list=romantic_vocab)

            famrel_vocab = [4, 5, 2, 1, 3]
            famrel_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="famrel", vocabulary_list=famrel_vocab)

            freetime_vocab = [3, 5, 4, 2, 1]
            freetime_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="freetime", vocabulary_list=freetime_vocab)

            goout_vocab = [3, 2, 5, 1, 4]
            goout_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="goout", vocabulary_list=goout_vocab)

            Dalc_vocab = [3, 2, 5, 1, 4]
            Dalc_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Dalc", vocabulary_list=Dalc_vocab)

            Walc_vocab = [1, 4, 2, 3, 5]
            Walc_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="Walc", vocabulary_list=Walc_vocab)

            health_vocab = [3, 5, 1, 4, 2]
            health_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="health", vocabulary_list=health_vocab)
            g1_vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            g1_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="G1", vocabulary_list=g1_vocab)
            g2_vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            g2_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key="G2", vocabulary_list=g2_vocab)
            # g3_vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            # g3_column = tf.feature_column.categorical_column_with_vocabulary_list(
            #     key="G3", vocabulary_list=g3_vocab)
            feature_columns = [
                # tf.feature_column.indicator_column(school_column),
                tf.feature_column.indicator_column(sex_column),
                tf.feature_column.indicator_column(age_column),
                # tf.feature_column.indicator_column(address_column),
                # tf.feature_column.indicator_column(famsize_column),
                tf.feature_column.indicator_column(Pstatus_column),
                tf.feature_column.indicator_column(Medu_column),
                tf.feature_column.indicator_column(Fedu_column),
                #  tf.feature_column.indicator_column(Mjob_column),
                #  tf.feature_column.indicator_column(Fjob_column),
                # tf.feature_column.indicator_column(reason_column),
                # tf.feature_column.indicator_column(guardian_column),
                tf.feature_column.indicator_column(traveltime_column),
                tf.feature_column.indicator_column(studytime_column),
                tf.feature_column.indicator_column(failures_column),
                tf.feature_column.indicator_column(schoolsup_column),
                tf.feature_column.indicator_column(famsup_column),
                # tf.feature_column.indicator_column(paid_column),
                # tf.feature_column.indicator_column(activities_column),
                # tf.feature_column.indicator_column(nursery_column),
                tf.feature_column.indicator_column(higher_column),
                tf.feature_column.indicator_column(internet_column),
                # tf.feature_column.indicator_column(romantic_column),
                tf.feature_column.indicator_column(famrel_column),
                tf.feature_column.indicator_column(freetime_column),
                tf.feature_column.indicator_column(goout_column),
                tf.feature_column.indicator_column(Dalc_column),
                tf.feature_column.indicator_column(Walc_column),
                tf.feature_column.indicator_column(health_column),
                tf.feature_column.numeric_column('absences'),
                tf.feature_column.indicator_column(g1_column),
                tf.feature_column.indicator_column(g2_column),
                # tf.feature_column.indicator_column(g3_column)
            ]
            input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100,
                                                                       num_epochs=10,
                                                                       shuffle=False)

            model = tf.estimator.DNNRegressor(hidden_units=[22, 22, 22], feature_columns=feature_columns,
                                              optimizer=tf.optimizers.Adam(learning_rate=0.01),
                                              activation_fn=tf.nn.relu)

            model.train(input_fn=input_func, steps=10000)

            predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
                x=X_test,
                batch_size=100,
                num_epochs=1,
                shuffle=False,
            )
            pred_gen = model.predict(predict_input_func)
            predictions = list(pred_gen)
            final_preds = []

            for pred in predictions:
                final_preds.append(pred['predictions'])

            from sklearn.metrics import mean_squared_error
            print(mean_squared_error(y_test, final_preds))

            from sklearn.metrics import mean_absolute_error
            print(mean_absolute_error(y_test, final_preds))

            from sklearn.metrics import r2_score
            print(r2_score(y_test, final_preds))

            list_pred = []
            for num in final_preds:
                list_pred.append(num[0])

            d = {'y_test': y_test, 'final_preds': list_pred}
            df = pd.DataFrame(data=d)
            df.round(2)[:10]

            #print(d)
            #print(df)

            
            data_edu = pd.read_csv('C:/Users/eduar/PycharmProjects/Projeto3/Projeto3/template/prever.csv')
            y_edu = data_edu.iloc[:, -1]
            data_edu = data_edu.drop(['G3'], axis=1)
            data_edu.head()
            data_edu['absences'] = scaler.fit_transform(data_edu['absences'].values.reshape(-1, 1))

            predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
                x=data_edu,
                batch_size=100,
                num_epochs=1,
                shuffle=False,
            )
            pred_gen = model.predict(predict_input_func)
            predictions = list(pred_gen)
            final_preds = []

            for pred in predictions:
                final_preds.append(pred['predictions'])

            list_pred = []
            valor=0
            for num in final_preds:
                list_pred.append(num[0])
                valor=num[0]
            #print('edu')
            #print(valor.round(2))  ## valor final

            #d = {'final_preds': list_pred}
            #df = pd.DataFrame(data=d)
            #df.round(2)[:10]
            #print(d)
            #print(df)

            #SVM
            # Import the dataset
            dataset = pd.read_csv(caminho, sep=",")
            X = dataset.iloc[:, [30, 31]].values
            y = dataset.iloc[:, -1].values
            data_pev = pd.read_csv('C:/Users/eduar/PycharmProjects/Projeto3/Projeto3/template/prever.csv', sep=",")
            u = data_pev.iloc[:, [30, 31]].values
            g = data_pev.iloc[:, -1].values

            print("Violin plot")
            sns.violinplot(dataset['sex'], dataset['G3'])
            sns.violinplot(data_pev['sex'], data_pev['G3'])
            sns.despine()

            #fig = plt.figure()
            #ax = fig.add_subplot(1, 1, 1)
            #ax.boxplot(dataset['G3'])
            #plt.show()




            def convert(School):
                if (School == 'GB'):
                    return 1
                else:
                    return 0

            dataset['school'] = dataset['school'].apply(convert)
            data_pev['school'] = data_pev['school'].apply(convert)


            def convert(g3):
                if (g3 >= 10):
                    return 1
                else:
                    return 0

            dataset['G3'] = dataset['G3'].apply(convert)
            data_pev['G3'] = data_pev['G3'].apply(convert)


            def yes_or_no(parameter):
                if parameter == 'yes':
                    return 1
                else:
                    return 0

            def yn(c):
                dataset[c] = dataset[c].apply(yes_or_no)
                data_pev[c] = data_pev[c].apply(yes_or_no)

            Colum = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

            for c in Colum:
                yn(c)


            X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=17)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_validation = sc.transform(X_validation)

            u = sc.transform(u)

            classifier = SVC(kernel='poly', degree=3, random_state=0)
            classifier.fit(X_train, y_train)


            prediction = classifier.predict(X_validation)
            prediction2 = classifier.predict(u)


            cmatrix = confusion_matrix(y_validation, prediction)
            print("=============================")
            print("Confusion matrix :")
            print(cmatrix)
            print("=============================")
            print("Accuracy: ", 100 * accuracy_score(y_validation, prediction), "%")
            print("=============================")
            print(classification_report(y_validation, prediction))
            print("=============================")


            X_set, y_set = X_validation, y_validation
            X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                           stop=X_set[:, 0].max() + 1, step=0.01),
                                 np.arange(start=X_set[:, 1].min() - 1,
                                           stop=X_set[:, 1].max() + 1, step=0.01))
            plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                         alpha=0.75, cmap=ListedColormap(('red', 'green')))
            #plt.xlim(X1.min(), X1.max())
            #plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                            c=ListedColormap(('yellow', 'blue'))(i), label=j)
            #plt.title('SVM (Student Prediction)')
            #plt.xlabel('G1')
            #plt.ylabel('G2')
            #plt.legend()
            #plt.show()

            def split(word):
                return [char for char in word]

            valor2 = []


            valor2 = split(prediction2)
            if valor2[0] ==0:
                svm = 'Insucesso académico'
            if valor2[0] ==1:
                svm = 'Sucesso académico'

            return HttpResponseRedirect('/prev_resultado/?name={}&nota={}&svm={}'.format(user,valor,svm))
        return render(request, 'principal.html', {'posts': user, 'string': string})
    return render(request,'principal.html',{'posts': user, 'string': string})
