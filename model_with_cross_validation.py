# -*- coding: utf-8 -*-
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
dados.head()

dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

x = dados[["preco", "idade_do_modelo","km_por_ano"]]
y = dados["vendido"]

SEED = 158020
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(treino_x, treino_y)
acuracia = dummy_stratified.score(teste_x, teste_y) * 100

print("A acurácia do dummy stratified foi de %.2f%%" % acuracia)

from sklearn.tree import DecisionTreeClassifier

SEED = 158020
np.random.seed(SEED)
modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print ("A acurácia foi %.2f%%" % acuracia)

x = dados[["preco", "idade_do_modelo","km_por_ano"]]
y = dados["vendido"]

SEED = 158020
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia do dummy stratified foi %.2f%%" % acuracia)

x = dados[["preco", "idade_do_modelo","km_por_ano"]]
y = dados["vendido"]

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia do dummy stratified foi %.2f%%" % acuracia)

from sklearn.model_selection import cross_validate

SEED = 158020

modelo = DecisionTreeClassifier(max_depth=2)

results = cross_validate(modelo,x,y, cv=3)

media = results['test_score'].mean()

desvio_padrao = results['test_score'].std()

print("Accuracy com cross validation 3 [%.2f %.2f]" % ((media -2*desvio_padrao)*100, (media +2*desvio_padrao)*100))

from sklearn.model_selection import cross_validate

SEED = 158020

modelo = DecisionTreeClassifier(max_depth=2)

results = cross_validate(modelo,x,y, cv=5)

media = results['test_score'].mean()

desvio_padrao = results['test_score'].std()

print("Accuracy com cross validation 3 [%.2f %.2f]" % ((media -2*desvio_padrao)*100, (media +2*desvio_padrao)*100))

"""#Aleatoriedade no cross validate

"""

def imprime_resultados (results):
  media = results['test_score'].mean()
  desvio_padrao = results['test_score'].std()
  print("Accuracy %.2f" %(media*100))
  print("Accuracy com cross validation [%.2f %.2f]" % ((media -2*desvio_padrao)*100, (media +2*desvio_padrao)*100))


imprime_resultados(results)

from sklearn.model_selection import KFold

SEED = 301

modelo = DecisionTreeClassifier(max_depth=2)

cv = KFold(n_splits=10)

results = cross_validate(modelo,x,y, cv=cv)

imprime_resultados(results)

from sklearn.model_selection import KFold

SEED = 301

modelo = DecisionTreeClassifier(max_depth=2)

cv = KFold(n_splits=10, shuffle=True)

results = cross_validate(modelo,x,y, cv=cv)

imprime_resultados(results)



"""# Simular situação de azar

"""

dados_azar = dados.sort_values("vendido",ascending=True)

x_azar = dados_azar[["preco", "idade_do_modelo", "km_por_ano"]]
y_azar = dados_azar["vendido"]

from sklearn.model_selection import KFold

SEED = 301

modelo = DecisionTreeClassifier(max_depth=2)

cv = KFold(n_splits=10,shuffle=True)

results = cross_validate(modelo,x_azar,y_azar, cv=cv, return_train_score=False)

imprime_resultados(results)

from sklearn.model_selection import StratifiedKFold

SEED = 301

modelo = DecisionTreeClassifier(max_depth=2)

cv = StratifiedKFold(n_splits=10,shuffle = True)

results = cross_validate(modelo,x_azar,y_azar, cv=cv, return_train_score=False)

imprime_resultados(results)

"""#Gerando dados aleatórios de modelo de carro para a simulação de agrupamento ao usar nosso estimador"""

np.random.seed(SEED)
dados['modelo']=dados.idade_do_modelo+np.random.randint(-2,3,size=10000)
dados.modelo=dados.modelo+abs(dados.modelo.min())+1
dados.head()

"""#Testando validação cruzada com GroupKFold"""

from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

modelo = DecisionTreeClassifier(max_depth=2)

cv = GroupKFold(n_splits=10)

results = cross_validate(modelo,x_azar,y_azar, cv=cv, groups=dados.modelo,return_train_score=False)

imprime_resultados(results)

"""#Cross validation com StandardScaler"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED=301
np.random.seed(SEED)

scaler=StandardScaler()

scaler.fit(treino_x)

treino_x_escalado=scaler.transform(treino_x)

teste_x_escalado =  scaler.transform(teste_x)

modelo = SVC()

modelo.fit(treino_x_escalado, treino_y)
previsoes = modelo.predict(teste_x_escalado)

acuracia = accuracy_score(teste_y, previsoes) * 100
print ("A acurácia foi %.2f%%" % acuracia)

from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

modelo = SVC()

cv = GroupKFold(n_splits=10)

results = cross_validate(modelo,x_azar,y_azar, cv=cv, groups=dados.modelo,return_train_score=False)

imprime_resultados(results)

scaler=StandardScaler()

scaler.fit(x_azar)

x_azar_escalado=scaler.transform(x_azar)

SEED = 301
np.random.seed(SEED)

modelo = SVC()

cv = GroupKFold(n_splits=10)

results = cross_validate(modelo,x_azar_escalado,y_azar, cv=cv, groups=dados.modelo,return_train_score=False)

imprime_resultados(results)

from sklearn.pipeline import Pipeline

SEED = 301
np.random.seed(SEED)

scaler = StandardScaler()
modelo= SVC()

pipeline = Pipeline([('transformacao',scaler),('estimador',modelo)])

cv = GroupKFold(n_splits=10)
results = cross_validate(pipeline,x_azar,y_azar, cv=cv, groups=dados.modelo,return_train_score=False)
imprime_resultados(results)

