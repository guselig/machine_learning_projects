#!/usr/bin/env python
# coding: utf-8

# ## PYTHON_ML_TRAINING_gradient_boosting_classifier_chemical_category
# 
# New notebook

# # **Importando as bibliotecas**

# In[1]:


#Importando as bibliotecas

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


# # **Configurando o experimento e importando a tabela do lakehouse**

# In[2]:


# Nomeando o experimento
mlflow.set_experiment("ML_EXPERIMENT_gradient_boosting_classifier_chemical_category")

# Dados do Lakehouse para o treinamento sem o product category 'Other Income'
vendas = spark.sql("SELECT * FROM LAKEHOUSE_DKBL.f_sales WHERE product_category <> 'Other Income'")
vendas = vendas.toPandas()

# Dados do Lakehouse para chemical_category
d_chemical_category = spark.sql("SELECT DISTINCT item_code,item_description, chemical_category FROM LAKEHOUSE_DKBL.d_products")
d_chemical_category = d_chemical_category.toPandas()


# # **Transformando o dataframe vendas #1**

# In[3]:


# Filtrando apenas produtos
vendas = vendas[vendas['item_code'].str[0]=="3"]

# Gerando a coluna 'item_code_bulk' em vendas
vendas['item_code_bulk'] = vendas['item_code'].str[0:6]

# Gerando a coluna 'item_code_bulk' em vendas
d_chemical_category['item_code_bulk'] = d_chemical_category['item_code'].str[0:6]

# Realizando o join de vendas com d_chemical_category
vendas = pd.merge(vendas,d_chemical_category[['item_code_bulk','chemical_category','item_description']].drop_duplicates(subset='item_code_bulk'),on ='item_code_bulk',how='left')

# Formatando as colunas que serão usadas de entrada do modelo
vendas[['chemical_category','item_description','customer_terminal']] = vendas[['chemical_category','item_description','customer_terminal']].apply(lambda x: x.str.upper())

# Contando quantas cada chemical_category aparece em vendas e transformando em um dataframe
drop_chemicals_categories = vendas['chemical_category'].value_counts().reset_index()
drop_chemicals_categories.columns = ['chemical_category', 'count']

# Filtrando as categorias que aparecem mais de 1 vez
drop_chemicals_categories = drop_chemicals_categories[drop_chemicals_categories['count']>1]

# Realizando o filtro no dataframe vendas
vendas = vendas[vendas['chemical_category'].isin(drop_chemicals_categories['chemical_category'])]

# Filling na em colunas numericas
vendas[['NET_invoice_value_BRL', 'order_quantity', 'unitary_cost']] = vendas[['NET_invoice_value_BRL', 'order_quantity', 'unitary_cost']].fillna(0)


# # **Transformando o dataframe vendas #2**

# In[4]:


# Adicionando as colunas de verificação do nome do item
def add_keyword_columns(df, column, keywords):

    for keyword in keywords:
        column_name = keyword.strip().lower().replace(" ", "_").replace("-", "_")
        df[column_name] = np.where(df[column].str.contains(keyword, na=False, case=False), True, False)
    return df

# Lista de palavras para busca
keywords = [
    "TYZOR", "TYZOR IAM", "TYZOR GBA", "MAXSCAV", "GASOLINA", "DIESEL", "ADITIVO", 
    "BREAKPLUS", "SOLVSCALE", "SYNTREAT", "COLDTOP", "SPEEDFLOW", "SR-", "UNICOR", 
    "FOAMTROM", "MAXFLOC", "DEFENDER", "ACTIFY", " DA ", "MPH ", "RETENMAX ", 
    "UCARSOL","UCARSOL AP","UCARSOL GT", "MILEX ", "OXITREAT", "RAPTOR", "CEPRO", "NEUTRACORR", "LEMONITE", 
    "DISPERSEPLUS", "EVOTHERM", "HYDRAFLO", "COOKINGMAX", "GUARDIAN", "CLEARLIQ", "DORF", "SOLVENT", "TRITON"
]

# Aplicando a função ao DataFrame vendas
vendas = add_keyword_columns(vendas, 'item_description', keywords)


# # **Treinando o modelo de ML**

# In[5]:


# Iniciando um job de treinanmento 
with mlflow.start_run() as run:

    #Definindo a SEED
    SEED = 301
    np.random.seed(SEED)

    # Declarando o StandardScaler
    scaler = StandardScaler()

    # Selecionando as colunas para padronização
    columns_to_scale = ['unitary_cost','order_quantity']

    # Aplicando o scaler e atualizando o DataFrame
    vendas[columns_to_scale] = scaler.fit_transform(vendas[columns_to_scale])    

    # Concatenando com os valores de custo
    xentrada = vendas[['unitary_cost','order_quantity', 
                        'tyzor','tyzor_iam', 'tyzor_gba', 'maxscav', 'gasolina', 'diesel', 'aditivo',
                        'breakplus', 'solvscale', 'syntreat', 'coldtop', 'speedflow', 'sr_','unicor', 
                        'foamtrom', 'maxfloc', 'defender', 'actify', 'da', 'mph','retenmax', 'ucarsol',
                        'ucarsol_ap', 'ucarsol_gt','milex', 'oxitreat', 'raptor', 'cepro','neutracorr', 
                        'lemonite', 'disperseplus', 'evotherm', 'hydraflo','cookingmax', 'guardian', 
                        'clearliq', 'dorf', 'solvent','triton']]

    #Dummizando a coluna y 'Product Category'
    yColumns = pd.get_dummies(vendas['chemical_category'])  

    #Dividindo o conjunto em treino e teste
    treinoX, testeX, treinoY, testeY = train_test_split(xentrada,yColumns,random_state = SEED, test_size = 0.25,stratify = yColumns)

    #Definindo o modelo de ML
    modelo = GradientBoostingClassifier(random_state = SEED,n_estimators=50, max_depth=3, learning_rate=0.1)

    #Treinando o modelo
    modelo.fit(treinoX, treinoY.idxmax(axis=1))  

    #Obtendo a acurácia
    previsoes = modelo.predict(testeX)
    score = accuracy_score(testeY.idxmax(axis=1), previsoes) * 100

    #Salvando as dimensões de entrada e saída do modelo treinado
    signature = infer_signature(treinoX, treinoY)

    #Ativando os logs do MLFlow para as métricas de treinamento
    print("test log_metrics.")
    mlflow.log_metric("score", score)

    #Registra o modelo treinado no MLflow
    print("test log_model.")
    mlflow.sklearn.log_model(modelo, "ML_MODEL_gradient_boosting_classifier_chemical_category", signature=signature)
    print("Model saved in run_id=%s" % run.info.run_id)
    
    #Registra o modelo no MLflow como um modelo específico que pode ser versionado e gerenciado.
    print("test register_model.")
    mlflow.register_model(
        "runs:/{}/ML_MODEL_gradient_boosting_classifier_chemical_category".format(run.info.run_id), "ML_MODEL_gradient_boosting_classifier_chemical_category"
    )
    
    print(f"All done, acurácia {score}")

