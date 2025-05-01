import pandas as pd
import numpy as np

data = pd.read_excel('/content/Database Rersources ML.xlsx')

data['Resposta'] = np.where(data['Item'].str[-6:] != '000000', "PLANNING" +", " + "FIX", "PLANNING")


data.head()

data['Bulk'] = np.where(data['Item'].str[-6:] == '000000', 1, 0)
data['Produto'] = np.where(data['Item'].str[0] == '3', 1, 0)
data['Intermediário'] = np.where(data['Item'].str[0] == '4', 1, 0)
data['Tyzor'] = np.where(data['Item Desc'].str.contains('TYZOR'), 1, 0)
data["Cepro"] = np.where(data['Item Desc'].str.contains('CEPRO'), 1, 0)
data['Terminação'] = np.where(data['Item'].str[-3:] != '000000', 1, 0)

data['WAREHOUSE'] = np.where(data['RESOURCES'].str.contains('WAREHOUSE'), 1, 0)
data['RELABEL'] = np.where(data['RESOURCES'].str.contains('RELABEL'), 1, 0)
data['RESALE'] = np.where(data['RESOURCES'].str.contains('RESALE'), 1, 0)
data['PALLET_CINTA'] = np.where(data['RESOURCES'].str.contains('PALLET'), 1, 0)


data.head()

x = data[['Bulk','Produto', 'Intermediário','Tyzor','Cepro','Terminação']]
y = data[['RELABEL',"WAREHOUSE","RESALE",'PALLET_CINTA']]

# Conta o número de linhas (listas internas) em x e y
num_linhas_x = len(x)
num_linhas_y = len(y)

print("Número de linhas em x:", num_linhas_x)
print("Número de linhas em y:", num_linhas_y)

from sklearn.dummy import DummyClassifier

dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(x_train, y_train)
acuracia = dummy_stratified.score(x_test, y_test) * 100

print("A acurácia do dummy stratified foi de %.2f%%" % acuracia)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def imprime_resultados (results):
  media = results['test_score'].mean()
  desvio_padrao = results['test_score'].std()
  print("Accuracy %.2f" %(media*100))
  print("Accuracy com cross validation [%.2f %.2f]" % ((media -2*desvio_padrao)*100, (media +2*desvio_padrao)*100))


  imprime_resultados(results)

"""#Logistic Regression"""

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from joblib import dump



SEED = 301

classifier = LogisticRegression(random_state=SEED)

modelo1 = MultiOutputClassifier(classifier)

cv = KFold(n_splits=3, shuffle=True)

results = cross_validate(modelo1,x,y, cv=cv)

print(results['test_score'])

imprime_resultados(results)

modelo1.fit(x_train,y_train)

"""#DecisionTreeClassifier"""

from sklearn.model_selection import GridSearchCV

seed=301
np.random.seed(SEED)

espaco_de_parametros = {
    "max_depth": [3,5],
    "min_samples_split":[32,64,128],
    "min_samples_leaf":[32,64,128],
    "criterion": ["gini","entropy"]
}

busca = GridSearchCV(DecisionTreeClassifier(),espaco_de_parametros,cv=KFold(n_splits=10,shuffle=True))

busca.fit(x,y)

melhores_parametros = busca.best_params_

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_validate

SEED = 301

classifier = DecisionTreeClassifier(**melhores_parametros,random_state=SEED)

modelo2 = MultiOutputClassifier(classifier)

cv = KFold(n_splits=3, shuffle=True)

results = cross_validate(modelo2, x, y, cv=cv)

print(results['test_score'])

imprime_resultados(results)

"""#Imputando os Dados"""

dados = pd.read_excel('/content/INPUT.xlsx')

dados['Saída1'] = np.where(dados['Item'].str[-6:] != '000000', "PLANNING" +", " + "FIX", "PLANNING")

dados.head()

dados['Bulk'] = np.where(dados['Item'].str[-6:] == '000000', 1, 0)
dados['Produto'] = np.where(dados['Item'].str[0] == '3', 1, 0)
dados['Intermediário'] = np.where(dados['Item'].str[0] == '4', 1, 0)
dados['Tyzor'] = np.where(dados['Item Desc'].str.contains('TYZOR'), 1, 0)
dados["Cepro"] = np.where(dados['Item Desc'].str.contains('CEPRO'), 1, 0)
dados["Terminação"] = np.where(dados['Item'].str[-3:] != '000000', 1, 0)


entrada = dados[['Bulk', 'Produto', 'Intermediário', 'Tyzor','Cepro','Terminação']]

entrada.head()

"""#Utilizando a Logistic Regression"""

resposta = modelo1.predict(entrada)

resposta = pd.DataFrame(resposta)

resposta['Saída2'] = resposta.apply(lambda row: ', '.join(filter(None, [
                                                    'RELABEL' if row[0] == 1 else '',
                                                    'WAREHOUSE' if row[1] == 1 else '',
                                                    'PALLET_CINTA' if row[3] == 1 else '',
                                                    'RESALE' if row[2] == 1 else ''])), axis=1)



resposta['Saída'] =  dados['Saída1'] +", " +resposta['Saída2']

resposta = pd.concat([dados[['Item', 'Item Desc']], resposta['Saída']], axis=1)

resposta



