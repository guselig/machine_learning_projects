# -*- coding: utf-8 -*-
"""Machine Learning: otimização de modelos através de hiperparâmetros

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_Apri6c8DC12w-hKB5l_ClwErxb7o8Sy
"""

!pip install graphviz ==0.9
!pip install pydot
!apt-get install graphviz

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

dados.head()

# situação horrível de "azar" onde as classes estão ordenadas por padrão

dados_azar = dados.sort_values("vendido", ascending=True)
x_azar = dados_azar[["preco", "idade_do_modelo","km_por_ano"]]
y_azar = dados_azar["vendido"]
dados_azar.head()

from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
import numpy as np

SEED = 301
np.random.seed(SEED)

modelo = DummyClassifier()
results = cross_validate(modelo, x_azar, y_azar, cv = 10, return_train_score=False)
media = results['test_score'].mean()
desvio_padrao = results['test_score'].std()
print("Accuracy com dummy stratified, 10 = [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))

from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = 301
np.random.seed(SEED)

modelo = DecisionTreeClassifier(max_depth=2)
results = cross_validate(modelo, x_azar, y_azar, cv = 10, return_train_score=False)
media = results['test_score'].mean()
desvio_padrao = results['test_score'].std()
print("Accuracy com cross validation, 10 = [%.2f, %.2f]" % ((media - 2 * desvio_padrao)*100, (media + 2 * desvio_padrao) * 100))

# gerando dados aleatórios de modelo de carro para simulação de agrupamento ao usar nosso estimador

np.random.seed(SEED)
dados['modelo'] = dados.idade_do_modelo + np.random.randint(-2, 3, size=10000)
dados.modelo = dados.modelo + abs(dados.modelo.min()) + 1
dados.head()

def imprime_resultados(results):
  media = results['test_score'].mean() * 100
  desvio = results['test_score'].std() * 100
  print("Accuracy médio %.2f" % media)
  print("Intervalo [%.2f, %.2f]" % (media - 2 * desvio, media + 2 * desvio))

# GroupKFold para analisar como o modelo se comporta com novos grupos

from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

cv = GroupKFold(n_splits = 10)
modelo = DecisionTreeClassifier(max_depth=2)
results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)

# GroupKFold em um pipeline com StandardScaler e SVC

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

SEED = 301
np.random.seed(SEED)

scaler = StandardScaler()
modelo = SVC()

pipeline = Pipeline([('transformacao',scaler), ('estimador',modelo)])

cv = GroupKFold(n_splits = 10)
results = cross_validate(pipeline, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)

from sklearn.tree import export_graphviz
import graphviz

modelo.fit(x_azar,y_azar)

features= x_azar.columns

dot_data=export_graphviz(modelo, out_file=None,filled=True,rounded=True,class_names=["não","sim"],feature_names=features)

graph=graphviz.Source(dot_data)

graph

from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

cv = GroupKFold(n_splits = 10)
modelo = DecisionTreeClassifier(max_depth=3)
results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)

from sklearn.tree import export_graphviz
import graphviz

modelo.fit(x_azar,y_azar)

features= x_azar.columns

dot_data=export_graphviz(modelo, out_file=None,filled=True,rounded=True,class_names=["não","sim"],feature_names=features)

graph=graphviz.Source(dot_data)

graph

from sklearn.model_selection import GroupKFold

SEED = 301
np.random.seed(SEED)

cv = GroupKFold(n_splits = 10)
modelo = DecisionTreeClassifier(max_depth=10)
results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=False)
imprime_resultados(results)

"""#Explorando hiperparâmetros em 1 dimensões"""

def roda_arvore_de_decisao(max_depth):

  SEED = 301
  np.random.seed(SEED)

  cv = GroupKFold(n_splits = 10)
  modelo = DecisionTreeClassifier(max_depth=max_depth)
  results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=True)
  train_score = results['train_score'].mean()*100
  test_score = results['test_score'].mean()*100
  print("Arvore max_depth = %d,treino=%.2f, teste=%.2f" % (max_depth,train_score,test_score))
  tabela = [max_depth,train_score, test_score]
  return tabela

resultados = [roda_arvore_de_decisao(i) for i in range (1,33)]
resultados = pd.DataFrame(resultados, columns = ['max_depth','train','test'])
resultados.head()

import seaborn as sns

sns.lineplot(x="max_depth",y = "train",data=resultados)

"""#OVERFIT: perfeito para o treino e ruim para o teste"""

import matplotlib.pyplot as plt


sns.lineplot(x="max_depth",y = "train",data=resultados)
sns.lineplot(x="max_depth",y = "test",data=resultados)
plt.legend(["Treino","Teste"])

resultados.sort_values("test",ascending=False).head()

"""#Explorando hiperparâmetros em duas dimensões"""

def roda_arvore_de_decisao(max_depth,min_samples_leaf):

  SEED = 301
  np.random.seed(SEED)

  cv = GroupKFold(n_splits = 10)
  modelo = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
  results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=True)
  train_score = results['train_score'].mean()*100
  test_score = results['test_score'].mean()*100
  print("Arvore max_depth = %d,treino=%.2f,min_samples_leaf=%d, teste=%.2f" % (max_depth,min_samples_leaf,train_score,test_score))
  tabela = [max_depth,min_samples_leaf,train_score, test_score]
  return tabela

def busca():
  resultados = []
  for max_depth in range (1,33):
    for min_samples_leaf in [32,64,128,256]:
      tabela = roda_arvore_de_decisao(max_depth,min_samples_leaf)
      resultados.append(tabela)
  resultados = pd.DataFrame(resultados, columns = ['max_depth','min_samples_leaf','train','test'])
  return resultados

resultados = busca()
resultados.head()

resultados.sort_values("test",ascending=False).head()

corr=resultados.corr()
corr

sns.heatmap(corr)

sns.pairplot(resultados)

def busca():
  resultados = []
  for max_depth in range (1,33):
    for min_samples_leaf in [128,192,256,512]:
      tabela = roda_arvore_de_decisao(max_depth,min_samples_leaf)
      resultados.append(tabela)
  resultados = pd.DataFrame(resultados, columns = ['max_depth','min_samples_leaf','train','test'])
  return resultados

resultados = busca()
resultados.head()

corr=resultados.corr()
corr

resultados.sort_values("test",ascending=False).head()

"""# Explorando 3 dimensões de hiperparâmetros"""

def roda_arvore_de_decisao(max_depth,min_samples_leaf, min_samples_split):

  SEED = 301
  np.random.seed(SEED)

  cv = GroupKFold(n_splits = 10)
  modelo = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
  results = cross_validate(modelo, x_azar, y_azar, cv = cv, groups = dados.modelo, return_train_score=True)
  fit_time = results['fit_time'].mean()
  score_time = results['score_time'].mean()
  train_score = results['train_score'].mean()*100
  test_score = results['test_score'].mean()*100
  tabela = [max_depth,min_samples_leaf,min_samples_split,train_score, test_score,fit_time,score_time]
  return tabela

def busca():
  resultados = []
  for max_depth in range (1,33):
    for min_samples_leaf in [32,64,128,256]:
      for min_samples_split in[32,64,128,256]:
        tabela = roda_arvore_de_decisao(max_depth,min_samples_leaf,min_samples_split)
        resultados.append(tabela)
  resultados = pd.DataFrame(resultados, columns = ['max_depth','min_samples_leaf','min_samples_split','train','test','fit_time','score_time'])
  return resultados

resultados = busca()

corr = resultados.corr()
corr

resultados.sort_values("test",ascending=False).head()

"""#Explorando espaço de hiperparâmetros com GrisSearch CV"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

seed=301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth": [3,5],
    "min_samples_split":[32,64,128],
    "min_samples_leaf":[32,64,128],
    "criterion": ["gini","entropy"]
}

busca = GridSearchCV(DecisionTreeClassifier(),espaco_de_parametros,cv=KFold(n_splits=10,shuffle=True))

busca.fit(x_azar,y_azar)

resultados = pd.DataFrame(busca.cv_results_)
resultados.head()

print(busca.best_params_)
print(busca.best_score_*100)

melhor=busca.best_estimator_

from sklearn.metrics import accuracy_score

#evitar essa abordagem pois estará sendo otimista

predicoes = melhor.predict(x_azar)

accuracy_score(predicoes,y_azar)*100

"""# Como ter uma estimativa sem esse vício nos dados que eu já vi?

No caso de cross validation com busca de hiper parâmetros, famos uma nova validação cruzada chamada nested cross validation.
"""

from sklearn.model_selection import cross_val_score

scores = cross_val_score(busca,x_azar,y_azar, cv=GroupKFold(n_splits=10),groups=dados.modelo)

"""# Infelizmente como o Pandas não suporta nested validarion com groupKFold vamos usar o KFold normal"""

from sklearn.model_selection import cross_val_score

scores = cross_val_score(busca,x_azar,y_azar, cv=KFold(n_splits=10,shuffle=True),groups=dados.modelo)

def imprime_scores(scores):
  media = scores.mean() * 100
  desvio = scores.std() * 100
  print("Accuracy médio %.2f" % media)
  print("Intervalo [%.2f, %.2f]" % (media - 2 * desvio, media + 2 * desvio))

imprime_scores(scores)

melhor=busca.best_estimator_

print(melhor)

from sklearn.tree import export_graphviz
import graphviz

features= x_azar.columns

dot_data=export_graphviz(melhor, out_file=None,filled=True,rounded=True,class_names=["não","sim"],feature_names=features)

graph=graphviz.Source(dot_data)

graph

