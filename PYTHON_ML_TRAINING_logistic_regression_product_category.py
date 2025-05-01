#!/usr/bin/env python
# coding: utf-8

# ## PYTHON_ML_TRAINING_logistic_regression_product_category
# 
# Training ML model to classify product category in xxdk_brazil_sales_mang_tbl table

# # **Importando as bibliotecas**

# In[1]:


#Importando as bibliotecas

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score


# # **Configurando o experimento e importando a tabela do lakehouse**

# In[4]:


# Nomeando o experimento
mlflow.set_experiment("ML_EXPERIMENT_logistic_regression_product_category")

# Dados do Lakehouse para o treinamento sem o product category 'Other Income'
vendas = spark.sql("SELECT * FROM LAKEHOUSE_DKBL.f_sales WHERE product_category <> 'Other Income' and product_category <> 'Othe Income' ")
vendas = vendas.toPandas()

# Dados do Lakehouse para chemical_category
d_chemical_category = spark.sql("SELECT DISTINCT item_code,item_description, chemical_category FROM LAKEHOUSE_DKBL.d_products")
d_chemical_category = d_chemical_category.toPandas()


# # **Treinando o modelo - ML_MODEL_logistic_regression_product_category**

# In[6]:


# Iniciando um job de treinanmento 
with mlflow.start_run() as run:

    #Definindo a SEED
    SEED = 301
    np.random.seed(SEED)

    #Colocando todas colunas de string em letras em maísculas
    vendas[['customer_master_sales_person','product_category']] = vendas[['customer_master_sales_person','product_category']].apply(lambda x: x.str.upper())

    #Gerando o dataframe das colunas dummies x
    xEntrada = pd.DataFrame(vendas['customer_master_sales_person'])
    xDummies = pd.get_dummies(xEntrada)

    #Salvando dummies x com valores unicos como uma delta table no lakehouse
    xEntrada_unique = pd.DataFrame({'customer_master_sales_person': xEntrada['customer_master_sales_person'].unique()})
    xEntrada_unique_spark = spark.createDataFrame(xEntrada_unique)
    xEntrada_unique_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("LAKEHOUSE_DKBL.ML_logistic_regression_product_category_x_entrada_unique")

    #Dummizando a coluna y 'Product Category'
    yColumns = pd.get_dummies(vendas['product_category'])  

    #Dividindo o conjunto em treino e teste
    treinoX, testeX, treinoY, testeY = train_test_split(xDummies,yColumns,random_state = SEED, test_size = 0.25,stratify = yColumns)

    #Definindo o modelo de ML
    modelo = LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000)

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
    mlflow.sklearn.log_model(modelo, "ML_MODEL_logistic_regression_product_category", signature=signature)
    print("Model saved in run_id=%s" % run.info.run_id)
    
    #Registra o modelo no MLflow como um modelo específico que pode ser versionado e gerenciado.
    print("test register_model.")
    mlflow.register_model(
        "runs:/{}/ML_MODEL_logistic_regression_product_category".format(run.info.run_id), "ML_MODEL_logistic_regression_product_category"
    )
    
    print(f"All done, acuracia {score}")

