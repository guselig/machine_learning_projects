import pandas as pd
import numpy as np
from datetime import datetime
import mlflow.sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

xxdk_us_ir_iso_t_gold = spark.sql("SELECT * FROM LAKEHOUSE_DK_US.xxdk_us_ir_iso_t_gold")
display(xxdk_us_ir_iso_t_gold)

f_xxdk_us_lotwise_inv_t_bi_hist = spark.sql("SELECT * FROM LAKEHOUSE_DK_US.f_xxdk_us_lotwise_inv_t_bi_hist")
display(f_xxdk_us_lotwise_inv_t_bi_hist)

# Transformando para dataframe Pandas
f_xxdk_us_lotwise_inv_t_bi_hist_pd = f_xxdk_us_lotwise_inv_t_bi_hist.toPandas()
xxdk_us_ir_iso_t_gold_pd = xxdk_us_ir_iso_t_gold.toPandas()

# Filtrando os itens em transito em xxdk_us_ir_iso_t_gold_pd
xxdk_us_ir_iso_t_gold_pd_transit = xxdk_us_ir_iso_t_gold_pd[(xxdk_us_ir_iso_t_gold_pd['receipt_date'].isnull()) & (xxdk_us_ir_iso_t_gold_pd['line_status']!="AWAITING_SHIPPING")]

# Realizando o filtro de data em xxdk_us_ir_iso_t_gold_pd para excluir os itens que nao tiveram o status em transito atualizado
filter = pd.to_datetime("01-01-2020",dayfirst=True)
xxdk_us_ir_iso_t_gold_pd_transit=xxdk_us_ir_iso_t_gold_pd_transit[pd.to_datetime(xxdk_us_ir_iso_t_gold_pd_transit['request_date'])>=filter]

# Filtrando o estoque atual e em transito em f_xxdk_us_lotwise_inv_t_bi_hist_pd
f_xxdk_us_lotwise_inv_t_bi_hist_pd = f_xxdk_us_lotwise_inv_t_bi_hist_pd[(f_xxdk_us_lotwise_inv_t_bi_hist_pd['actual_stock']==True) & (f_xxdk_us_lotwise_inv_t_bi_hist_pd['movement_status']=="In Transit")]

# Convertendo as colunas 'actual_ship_date' para datetime nas duas DataFrames
f_xxdk_us_lotwise_inv_t_bi_hist_pd['intransit_date'] = pd.to_datetime(f_xxdk_us_lotwise_inv_t_bi_hist_pd['intransit_date'])
xxdk_us_ir_iso_t_gold_pd_transit['actual_ship_date'] = pd.to_datetime(xxdk_us_ir_iso_t_gold_pd_transit['actual_ship_date'])

# Cruzando o f_xxdk_us_lotwise_inv_t_bi_hist_pd com xxdk_us_ir_iso_t_gold_pd_transit para encontrar o organization source e requistion date de cada lot em transito
f_xxdk_us_lotwise_inv_t_bi_hist_pd = pd.merge(f_xxdk_us_lotwise_inv_t_bi_hist_pd,xxdk_us_ir_iso_t_gold_pd_transit[['item','requisition_date','actual_ship_date','source_organization_code','organization_code']],left_on=('item','organization_code','intransit_date'),right_on=('item','organization_code','actual_ship_date'),how='left')

# Cruzando o f_xxdk_us_lotwise_inv_t_bi_hist_pd com xxdk_us_ir_iso_t_gold_pd_transit pela segunda vez para porem sem usar instransit_date e actual_ship_date como parametro
f_xxdk_us_lotwise_inv_t_bi_hist_pd = pd.merge(f_xxdk_us_lotwise_inv_t_bi_hist_pd,xxdk_us_ir_iso_t_gold_pd_transit[['item','source_organization_code','organization_code']],left_on=('item','organization_code'),right_on=('item','organization_code'),how='left')

# Puxar o intransit_date quando requisition_date for nulo
f_xxdk_us_lotwise_inv_t_bi_hist_pd['requisition_date'] = np.where(f_xxdk_us_lotwise_inv_t_bi_hist_pd['requisition_date'].isnull(),f_xxdk_us_lotwise_inv_t_bi_hist_pd['intransit_date'],f_xxdk_us_lotwise_inv_t_bi_hist_pd['requisition_date'])

# Puxar o source_organization_code_y quando o source_organization_code_x for nulo
f_xxdk_us_lotwise_inv_t_bi_hist_pd['source_organization_code'] = np.where(f_xxdk_us_lotwise_inv_t_bi_hist_pd['source_organization_code_x'].isnull(),f_xxdk_us_lotwise_inv_t_bi_hist_pd['source_organization_code_y'],f_xxdk_us_lotwise_inv_t_bi_hist_pd['source_organization_code_x'])
f_xxdk_us_lotwise_inv_t_bi_hist_pd = f_xxdk_us_lotwise_inv_t_bi_hist_pd.drop(columns=['source_organization_code_x','source_organization_code_y'])



# Selecionando as colunas para treinar o modelo
xxdk_us_ir_iso_t_gold_pd = xxdk_us_ir_iso_t_gold_pd[['requisition_date','source_organization_code','organization_code','line_status','receipt_date','shipping_method']]

# Obter a data atual
today = pd.to_datetime(datetime.today())

# Subtrair dois anos da data atual
two_years_ago = today - pd.DateOffset(month=6)

# Filtrando apenas os fretes por mar
xxdk_us_ir_iso_t_gold_pd = xxdk_us_ir_iso_t_gold_pd[xxdk_us_ir_iso_t_gold_pd['shipping_method']=="Sea"]

# Exlcuindo os itens que ainda estao em transito
xxdk_us_ir_iso_t_gold_pd = xxdk_us_ir_iso_t_gold_pd[~xxdk_us_ir_iso_t_gold_pd['receipt_date'].isnull()]

# Realizando o filtro de data
xxdk_us_ir_iso_t_gold_pd = xxdk_us_ir_iso_t_gold_pd[xxdk_us_ir_iso_t_gold_pd['requisition_date']>=two_years_ago]

# Calculando o tempo de lead time
xxdk_us_ir_iso_t_gold_pd['lead_time'] = (xxdk_us_ir_iso_t_gold_pd['receipt_date'] - xxdk_us_ir_iso_t_gold_pd['requisition_date']).dt.days


# Definindo a SEED
SEED = 301
np.random.seed(SEED)

# Criando as variáveis dummies para X 
x = pd.get_dummies(xxdk_us_ir_iso_t_gold_pd[['source_organization_code', 'organization_code','business_line']])

# Definindo a variável alvo y (lead time)
y = xxdk_us_ir_iso_t_gold_pd[['lead_time']]

# Aplicando escalonamento em y
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Definindo o modelo
model = RandomForestRegressor(random_state=SEED)

# Criando os scorers personalizados
scoring = {
    'MAE': make_scorer(mean_absolute_error),
    'R2': make_scorer(r2_score)
}

# Realizando cross-validation (3 folds)
results = cross_validate(model, x, y_scaled.ravel(), cv=3, scoring=scoring, return_train_score=False)

# Exibindo os resultados
print(f"MAE Médio: {np.mean(results['test_MAE'])}")
print(f"R² Médio: {np.mean(results['test_R2'])}")