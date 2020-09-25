## ReferÊncias:
# Joins (Merge) com Pandas: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
# Exemplos de gráficos de barras: https://www.tradingcomdados.com/post/analisando-dados-fundamentalistas-com-python
# Seaborn Boxplot: https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot
# Catplot Seaborn: https://seaborn.pydata.org/generated/seaborn.catplot.html
from itertools import count

import pandas as pd
import seaborn as sns
from altair.vega import textValue
from matplotlib import pyplot as plt
import numpy as np
import pickle

# Configurações de plotagem dos gráficos:
from plotly.validators.carpet.aaxis import _linewidth

plt.style.use('ggplot')

small_size = 12
medium_size = 14
bigger_size = 16

plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=medium_size)  # fontsize of the figure title

#Clientes_base_hclust_clusterizado.drop(['CNPJ_x', 'level_0'], axis=1, inplace=True) # Deleta as colunas que não são necessárias para clusterização
#Clientes_base_hclust_clusterizado.rename(columns={'CNPJ_y':'CNPJ'}, inplace=True)

#%%
# ############################################################################################
# Dataset vindo do arquivo "1.3.Clusterizacao.py"
# ############################################################################################
# Join para buscar DataFrame original, filtrando apenas CNPJs clusterizados e trazendo o nº do cluster
# Clientes_clusterizado = pd.merge(left=Clientes_threshold,
#                                  right=Clientes_base_hclust_clusterizado[["CNPJ","Cluster"]],
#                                  how='inner',
#                                  on='CNPJ',
#                                  left_index=False,
#                                  right_index=False)

#%%
# --------------------------------------------------------------------------------------------
# Análise exploratória dos Clusteres:
# 1. Variáveis numéricas - Boxplot das variáveis por Cluster
#   X = cluster
#   y = variáveis
# --------------------------------------------------------------------------------------------

# Converter coluna de cluster em string:
Clientes_clusterizado['Cluster'] = Clientes_clusterizado['Cluster'].astype(str)

# Renomear os "clusteres" para grupos, de Grupo-1 até Grupo-N
Clientes_clusterizado['Cluster'] = Clientes_clusterizado['Cluster'].replace(['0','1','2','3','4','5','6'],
                                                                            ['Grupo-1','Grupo-2','Grupo-3','Grupo-4','Grupo-5','Grupo-6','Grupo-7'])

# Ordenar dataframe por cluster de Grupo-1 até Grupo-N
Clientes_clusterizado = Clientes_clusterizado.sort_values(by=['Cluster'], ascending=True)

# Instanciar objeto das figurar para plotar gráficos:
fig = 0

#%%
# --------------------------------------------------------------------------------------------
# 2. Variáveis Categóricas:
# - Comparação do volume (count de CNPJs) em gráfico de barras
# --------------------------------------------------------------------------------------------

# Variáveis que precisarão ser analisadas por cluster, mas com o % em relação ao total. Comparar o % da base full Vs o % do cluster.
# Tipo_cadastro
# Atividade
# Segmento
# Natureza_Juridica
# Regime_Tributario

# Distribução da base de clientes clusterizada: tamanho dos clusteres
Cluster_distr = Clientes_clusterizado['CNPJ'].groupby(Clientes_clusterizado['Cluster']).count()
Cluster_distr = pd.DataFrame(Cluster_distr).reset_index()
Cluster_distr['Perc'] = Cluster_distr['CNPJ']/Cluster_distr['CNPJ'].sum()

#%%
# Plotar gráfico com a distribuição dos custeres:

# Título do gráfico
#title = 'Distribuição dos CNPJs entre os clusteres'

plt.rcParams.update({'font.size': 18})

fig_chart = plt.figure(fig+1, figsize=(15, 10))
ax = fig_chart.add_subplot(111)

# Gráfico de barras
bar_x = Cluster_distr['Cluster']
bar_y = Cluster_distr['CNPJ']
bar_label = 'Qtde CNPJs'

ax.bar(x=bar_x,
       height=bar_y,
       width=0.8, # Distrância entre as barras (padrão 0.8)
       linewidth=0, # Sem bordas nas barras
       color='blue',
       label=bar_label)
ax.grid(False)
ax.set_xticks(bar_x)
#ax.set_title(title, fontsize=16)
ax.set_xlabel('', c='black')
ax.set_ylabel(bar_label, fontsize=20, c='black')
ax.legend(loc='upper center')

# Adicionar label nos valores das barras:
for a,b in bar_y.items():
    ax.text(a, b+ 50, # Posição X e Y do label
            str(b), # isso é o que será plotado
            color='blue',
            fontweight='bold',
            horizontalalignment='center',
            weight='bold',
            size=18)
#-----------------------------------------------------------------
# Gráfico de linhas:
ax2 = ax.twinx() # dividir eixo Y

line_y = Cluster_distr['Perc']
line_label = '% do total'

ax2.plot(line_y,
         label=line_label,
         linewidth=3,
         color='red',
         marker='o',
         markersize=10)
ax2.grid(False)
ax2.set_yticks(np.arange(0.0, 1.0, 0.1))
ax2.set_ylabel(line_label, fontsize=20, c='black')
ax2.legend(loc='upper right')

# Adicionar label nos valores das linhas:
for i,j in round(line_y, 4).items():
    #print(i) # eixo X
    #print(j) # eixo Y
    ax2.text(i, j+ 0.03, # Posição X e Y do label
            str(round(j*100,2))+' %', # isso é o que será plotado
            color='yellow',
            fontweight='bold',
            horizontalalignment='center',
            weight='bold',
            size=18
            #verticalalignment = 'top'
            )

plt.tight_layout()
plt.show()
