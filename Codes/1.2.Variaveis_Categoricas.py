## Esse script trata as variáveis categóricas do Dataset para poderem ser utilizadas na clusterização.

#%%
import pandas as pd

# ############################################################################################
# 1. Consulta dos dados e tratamento das varíaveis categóricas
# ############################################################################################

# --------------------------------------------------------------------------------------------
# Consulta no BigQuery
# --------------------------------------------------------------------------------------------
project_id = 'XXXX'

# --------------------------------------------------------------------------------------------
# Variáveis categóricas para serem transformadas:
# --------------------------------------------------------------------------------------------
# Tipo_cadastro
# Atividade
# Segmento
# Natureza_Juridica
# Regime_Tributario

# Variáveis categóricas do dataset:
Var_cat = ['Tipo_cadastro', 'Atividade', 'Segmento', 'Natureza_Juridica', 'Regime_Tributario']

# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# Método 1: One Hot Encoding
# --------------------------------------------------------------------------------------------
# Transformar as variáveis categóricas em classes dummy. Cada valor de cada variável categórica vira uma coluna com 0 e 1.
# Clientes_dummies = pd.get_dummies(Clientes, columns=Var_cat, drop_first=False)

### Notas:
#   O dataset com as dummies aumentou a dimensionalidade expressivamente. De 20 colunas para 251.
#   Para reduzir a dimensionalidade do dataset (maldição da dimensionalidade) e, também, para evitar que ocorram clusteres de 1 ou quase um único cliente,
#   utilizou-se o treshold de analisar 80% da base de clientes, o que reduz as combinações de classes das variáveis categóricas de 3.735 para 511 combinaçoẽs possíveis (redução de 86% no volume de combinações)

# --------------------------------------------------------------------------------------------
# Dataset clientes threshold: 80% da base
# --------------------------------------------------------------------------------------------
query2 = """ select * from TABLE """
data2 = pd.read_gbq(query=query2, project_id=project_id, dialect='standard', use_bqstorage_api=True)
Clientes_threshold = data2.copy()
#%%
Clientes_threshold_dummies = pd.get_dummies(Clientes_threshold, columns=Var_cat, drop_first=False)

### Notas:
#   A partir do dataset com o threshold de 80%, a dimensionalidade das colunas, em relação ao dataset original, foi de 20 para 86, porém reduzindo significativamente em relação as 251 colunas do dataset sem considerar o threshold.
#   Optou-se por seguir então com a clusterização no método One Hot Encoding a partir do dataset considerando o threshold de 80%


# ############################################################################################
# 2. Clusterização (hclust) por método "One Hot Encoding"
# ############################################################################################

