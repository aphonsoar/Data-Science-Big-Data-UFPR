## Referências:
# Exemplo de uso de agrupamento hierarquico com Dedrograma: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
# Tutorial básico de Dendrograma com limite "P": https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
# Linkage Matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#   Perform hierarchical/agglomerative clustering: scipy.cluster.hierarchy.linkage
# K-means e Elbow Method: https://predictivehacks.com/k-means-elbow-method-code-for-python/#:~:text=The%20Elbow%20method%20is%20a,its%20assigned%20center(distortions).
# Joins (Merge) com Pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

import pandas as pd
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from google.cloud import bigquery
import seaborn as sns

# Configurações de plotagem dos gráficos:
plt.style.use('ggplot')

small_size = 8
medium_size = 10
bigger_size = 12

plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=medium_size)  # fontsize of the figure title


# ############################################################################################
# Preparação do Dataset para clusterizar:
# ############################################################################################
# Dataset vindo do arquivo "1.2.Variaveis_Categoricas.py"
# Cria base para clusterizar
Clientes_base_hclust = Clientes_threshold_dummies

#%%
# # --------------------------------------------------------------------------------------------
# # Visualizar colunas que tenham "missing values": valor 'null' (NaN)
# # --------------------------------------------------------------------------------------------
# plt.figure(0, figsize=(10, 20))
# sns.set(font_scale=0.8)
# sns.heatmap(Clientes_threshold_dummies.isnull(), cbar=False) #cbar é a legenda
# plt.xlabel('Variáveis')
# plt.ylabel('Nº registro')
# plt.title('Análise de "missing values" (NaN)')
# plt.tight_layout()
# plt.show()

# Desconsidera os registros que tenham algum valor 'null' (NaN)
Clientes_base_hclust = Clientes_threshold_dummies.dropna()
# Leva o índice para a base
Clientes_base_hclust = Clientes_base_hclust.reset_index()
# Novo shape com dummies, índice e sem os NaN (21405, 51)
Clientes_base_hclust.shape
# Guardar o índice e CNPJs para juntar com a base clusterizada
Index_CNPJ = Clientes_base_hclust["CNPJ"]
# Retirar o CNPJ para não influenciar na clusterização. A chave do registro é o índice do DataFrame.
del Clientes_base_hclust["CNPJ"]
# Deletar índice antigo
del Clientes_base_hclust["index"]

#%%
# ############################################################################################
# CLUSTERIZAÇÃO
# ############################################################################################
# --------------------------------------------------------------------------------------------
# Dendrograma com a base de clientes completa
# --------------------------------------------------------------------------------------------
# Função para levar o valor da distância euclidiana para cada nó do dendrograma
def augmented_dendrogram(*args, **kwargs):

    ddata = shc.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
    return ddata

# Generate the linkage matrix: matriz com o cálculo das distâncias (base do dendrograma, cfe referência "Linkage Matrix")
Z = shc.linkage(Clientes_base_hclust, 'ward')

#%%
# --------------------------------------------------------------------------------------------
# Etapa 1: plotar o dendrograma para definir o nº de clusters.
# --------------------------------------------------------------------------------------------
plt.figure(2, figsize=(15, 10))
#plt.title("Dendrograma de clientes", size=12)
#shc.dendrogram( ## Dendrograma original
augmented_dendrogram( ## Dendrograma da função acima
    Z,
    truncate_mode='lastp', # trunca mostrando apenas os últimos "P" clusters
    p=50, # numero "P" de clusters, confrome o truncate_mode
    #show_leaf_counts=True,
    #show_contracted=True
)

# Linha p/ 7 clusteres
N_Clusters = 7 # Valores para 8 clusters
Linha_cima = 392
Linha_baixo = 346
Distancia1 = Linha_cima - Linha_baixo
Metade_distancia = Distancia1 / 2
Linha_horizontal = Linha_baixo + Metade_distancia
plt.axhline(linestyle='--', y=Linha_horizontal, color='blue')

plt.ylabel('')
plt.yticks([])
#plt.yticks(size=20)
plt.xticks(size=20, c='black')

plt.tight_layout()
plt.show()

#%%
# --------------------------------------------------------------------------------------------
# Traçar linha de corte no Dendrograma
# --------------------------------------------------------------------------------------------
N_Clusters = 7 # Valores para 8 clusters
Linha_cima = 392
Linha_baixo = 346
Distancia1 = Linha_cima - Linha_baixo
Metade_distancia = Distancia1 / 2
Linha_horizontal = Linha_baixo + Metade_distancia
plt.axhline(linestyle='--', y=Linha_horizontal, color='blue')

#%%
# --------------------------------------------------------------------------------------------
# K-Means e Elbow Method
# --------------------------------------------------------------------------------------------
distortions = []
K = range(1,50) # Número de clusteres para o K-means
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(Clientes_base_hclust)
    distortions.append(kmeanModel.inertia_)
#%%
# --------------------------------------------------------------------------------------------
# Plotando o Elbow chart (plotting the distortions of K-Means)
# --------------------------------------------------------------------------------------------
plt.figure(3, figsize=(15, 10))
plt.plot(K, distortions, 'bx-')
#plt.xlabel('Nº de clusters')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k', size=12)
plt.tight_layout()

# plt.ylabel('')
# plt.yticks([])
plt.xticks(K, size=16, c='black')
plt.yticks(distortions, size=16, c='black')

# --------------------------------------------------------------------------------------------
# Traçar linha de corte no Elbow chart
# --------------------------------------------------------------------------------------------
N_Clusters = N_Clusters # Nº de clusteres
plt.axvline(linestyle='--', # linha vertical
            x=N_Clusters,
            ymin=0,
            ymax=0.3,
            color='r')
plt.axhline(linestyle='--', # linha horizontal
            y=distortions[N_Clusters-1],
            xmin=0,
            xmax=0.3,
            color='r')

#plt.yticks(size=12)
plt.show()
#%%
# --------------------------------------------------------------------------------------------
# Cálculo de redução das distorções do K-Means e Elbow chart
# --------------------------------------------------------------------------------------------
# Cria o dataframe a partir das distorções calculadas pelo K-Means
distortions_df = pd.DataFrame(distortions)
# Identifica o tamanho do DataFrame
LenDistortion = distortions_df.shape[0]
# Cria coluna para popular a variação percentual da distorção
distortions_df['Var_percentual'] = np.nan
# Renomeia colunas do DataFrame
distortions_df.columns = ["Distortion", 'Var_percentual']

# Cálculo da variação percentual
for d in range(1, LenDistortion):
    distortions_df['Var_percentual'].iloc[d] = (distortions_df["Distortion"].iloc[d] / distortions_df["Distortion"].iloc[d-1]) - 1

# Plotando o gráfico de redução da distorção do Elbow chart (plotting the distortions of K-Means)
plt.figure(4, figsize=(15, 10))
plt.plot(distortions_df['Var_percentual'], 'bx-')
plt.xlabel('Nº de clusters')
plt.ylabel('Queda da distorção')
plt.title('Redução da distorção a cada novo cluster', size=12)
plt.tight_layout()
plt.xticks(K)

plt.axvline(linestyle='--', # linha vertical
            x=N_Clusters,
            ymin=0,
            ymax=0.95,
            color='r')

plt.yticks(size=12)
plt.xticks(size=12)
plt.show()

#%%
# --------------------------------------------------------------------------------------------
# Etapa 2: sabendo o nº de clusters, a partir do Dendrograma e Elbow chart, agrupar os registros em seus respectivos clusters
# --------------------------------------------------------------------------------------------
# Revisar o nº de clusteres
N_Clusters = 7

# Clusterização
cluster = AgglomerativeClustering(n_clusters=N_Clusters, affinity='euclidean', linkage='ward')
dados_clusters = cluster.fit_predict(Clientes_base_hclust)

# Transformando o array de dados clusterizados em DataFrame
dados_clusters = pd.DataFrame(dados_clusters)
# Renomeando a coluna do cluster
dados_clusters.columns = ['Cluster']

# Tratando o índice (PK) da base para fazer o join com o DataFrame clusterizado
Clientes_base_hclust_clusterizado = Clientes_base_hclust.reset_index()
del Clientes_base_hclust_clusterizado["index"]

# Join da base com o DataFrame clusterizado para levar o cluster para a base principal
Clientes_base_hclust_clusterizado = pd.merge(left=dados_clusters,
                                             right=Clientes_base_hclust_clusterizado,
                                             how='inner',
                                             left_index=True,
                                             right_index=True)

# Join para trazer o CNPJ para a base clusterizada
Clientes_base_hclust_clusterizado = pd.merge(left=Index_CNPJ,
                                             right=Clientes_base_hclust_clusterizado,
                                             how='inner',
                                             left_index=True,
                                             right_index=True)