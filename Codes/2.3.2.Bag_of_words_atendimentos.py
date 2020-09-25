import pickle
from google.cloud import bigquery
import pandas as pd
import nltk
import numpy as np
import urllib.request
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

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

#%%
# --------------------------------------------------------------------------------------------
# Consulta no BigQuery
# --------------------------------------------------------------------------------------------

# Tickets:
query_tickets = """ select * from TICKETS """
data_tickets = pd.read_gbq(query_tickets, project_id=project_id, dialect='standard', use_bqstorage_api=True)

# Chats:
query_chats = """ select * from CHATS """
data_chats = pd.read_gbq(query_chats, project_id=project_id, dialect='standard', use_bqstorage_api=True)

atendimentos = pd.concat([data_tickets, data_chats], ignore_index=True)

#%%
# ------------------------------------------------------------------------------------------------------------
# Stop words
# ------------------------------------------------------------------------------------------------------------
from nltk.corpus import stopwords
from string import punctuation

# Stopwords adicionais criadas a partir de arquivo:
stopwords_pt_aux = pd.read_csv("https://raw.githubusercontent.com/aphonsoar/Data-Science-Big-Data-UFPR/master/Codes/stopwords_tickets_chats.txt") # Github

# Stopwords e pontuação em português vindas da biblioteca:
stopwords_pt_lib = set(stopwords.words('portuguese') + list(punctuation))

# Converter stopwords da lib para DF:
stopwords_pt_lib = pd.DataFrame(stopwords_pt_lib)
stopwords_pt_lib.columns = ['words']

# Juntar bases de stopwords:
stopwords_pt = pd.concat([stopwords_pt_aux,stopwords_pt_lib], ignore_index=True)

# Remover duplicados:
stopwords_pt = stopwords_pt.words.unique()
# Transformar objeto em lista
stopwords_pt = stopwords_pt.tolist()

stopwords_pt_df = pd.DataFrame(stopwords_pt)
stopwords_pt_df.columns = ['words']
#%%
# ------------------------------------------------------------------------------------------------------------
# Limpeza e tratamento para construir o bag of words
# ------------------------------------------------------------------------------------------------------------
# Amostra do dataset para teste:
atendimentos_sample_original = atendimentos#.sample(n=1000, random_state=3) #random_state é o seed #Amostra

# Criar DataFrame para cruzar com o índice dos "Bag of Words" das amostras:
atendimentos_sample_df_bag_of_words = atendimentos_sample_original.reset_index()

# Converter o texto dos atendimentos em lista:
atendimentos_sample = atendimentos_sample_original["Texto"].tolist()

# Converter Series para string:
#atendimentos_sample = pd.Series(atendimentos_sample["Texto"]).to_string()

# Limpeza e tratamentos: In the script above, we iterate through each sentence in the corpus, , and then remove the  and  from the text.
#Time
limpeza_start_time = time.time()

for i in range(len(atendimentos_sample)):
    atendimentos_sample[i] = atendimentos_sample[i].lower() # convert the sentence to lower case
    atendimentos_sample[i] = re.sub(r'\W',' ',atendimentos_sample[i]) # remove the punctuation
    atendimentos_sample[i] = re.sub(r'\s+',' ',atendimentos_sample[i]) # remove empty spaces
    atendimentos_sample[i] = re.sub('(\d)+','',atendimentos_sample[i]) # retira todos os números da string

limpeza_end_time = time.time()
Tempo_limpeza = round((limpeza_end_time - limpeza_start_time))
print('Tempo_limpeza: ' + str(Tempo_limpeza))

#%%
# ------------------------------------------------------------------------------------------------------------
# Identificar e analisar as palavras que mais se repetem
# ------------------------------------------------------------------------------------------------------------
# Time:
id_palavras_start_time = time.time()

wordfreq = {} # we created a dictionary called wordfreq.
for atendimento in atendimentos_sample: # Next, we iterate through each sentence in the corpus. The sentence is tokenized into words.
    tokens = nltk.word_tokenize(atendimento)
    for token in tokens: # Next, we iterate through each word in the sentence.
        if token not in stopwords_pt: # IF adicional incluído para desconsiderar stopwords >> Retirar stopwords e pontuação, etc
            if token not in wordfreq.keys(): # Se for a primeira vez que a palavra aparece:
                wordfreq[token] = 1 # recebe valor 1
            else:
                wordfreq[token] += 1 # se a palavra já apareceu antes, soma +1. Ou seja, é um count de vezes que a palavra aparece.

# Palavras que mais se repetem:
wordfreq_df = pd.Series(wordfreq).to_frame()
wordfreq_df.columns = ['freq']
wordfreq_df = wordfreq_df.sort_values(by='freq', ascending=False)

# Calcular o percentual de cada linha e acumulado no DataFrame:
wordfreq_df['perc'] = wordfreq_df['freq']/wordfreq_df['freq'].sum()
wordfreq_df['perc_acum'] = wordfreq_df['perc'].cumsum()
wordfreq_df = wordfreq_df.reset_index()

# Renomear a coluna da palavra:
wordfreq_df.rename(columns={'index':'word'}, inplace=True)

# Total de palavras distintas
wordfreq_df.head(10)
wordfreq_df.shape # 95.227 Palavras distintas

#%%
# ------------------------------------------------------------------------------------------------------------
# Histograma de distribuição das palavras:
# ------------------------------------------------------------------------------------------------------------

#wordfreq_df = pd.read_csv("/home/aphonsoamaral/aphonso/DSBD-TCC-Arquivos/wordfreq_df_full.csv", sep=';')
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels

head=1000

fig = plt.figure(2, figsize=(15, 10))
ax = fig.add_subplot(111)

ax.bar(x=wordfreq_df['word'].head(head),
       height=wordfreq_df['freq'].head(head),
       width=1, # Sem distância entre as barras
       linewidth=0, # Sem bordas nas barras
       label='Histograma')
ax.set_xticks('')
#ax.set_title('Ocorrência decrescente das ' + str(head) + ' primeiras palavras que mais aparecem', fontsize=16)
ax.set_xlabel('')
ax.set_ylabel('Frequência', fontsize=25, c='black')
ax.legend(loc='upper left', prop={'size': 16})

ax2 = ax.twinx()
ax2.plot(wordfreq_df['perc_acum'].head(head),
         color='red',
         label='% acumulado')
ax2.set_yticks(np.arange(0.0, 1.0, 0.05))
ax2.set_ylabel('% acumulado', fontsize=25, c='black')
ax2.legend(loc='upper right', prop={'size': 16})

plt.tight_layout()
plt.show()

# Time:
id_palavras_end_time = time.time()
Tempo_id_palavras = round((id_palavras_end_time - id_palavras_start_time))
print('Tempo_id_palavras: ' + str(Tempo_id_palavras))
#%%
# ------------------------------------------------------------------------------------------------------------
# Análise sobre a ocorrência das palavras:
print('Qtde palavras: 95% total = ' + str(wordfreq_df['freq'][wordfreq_df["perc_acum"] <= 0.95].count()))
print('Qtde palavras: 99% total = ' + str(wordfreq_df['freq'][wordfreq_df["perc_acum"] <= 0.99].count()))
print('Qtde palavras + 50 rep = ' + str(wordfreq_df['freq'][wordfreq_df["freq"] >= 49].count()))
print('Qtde palavras + 10 rep = ' + str(wordfreq_df['freq'][wordfreq_df["freq"] >= 9].count()))
print('Qtde palavras apenas 1 rep = ' + str(wordfreq_df['freq'][wordfreq_df["freq"] == 1].count()))
print('Qtde vezes palavra: ' + str(wordfreq_df['freq'][wordfreq_df["word"] == 'aa']))

# Obter a frequência na posição:
wordfreq_df.iloc[5654]

#%%
# Exportar o DataFrame de frequência das palavras que mais se repetem: base full.
wordfreq_df_limpo = wordfreq_df.iloc[:5654+1]
pd.DataFrame.to_csv(wordfreq_df_limpo, sep=';', path_or_buf='/home/aphonsoamaral/files/wordfreq_df_limpo.csv', index=False)

# -- Conclusão: trabalhar apenas as top 95% de frequência, ou seja: 5654 palavras.

#%%
# ------------------------------------------------------------------------------------------------------------
# Retirar as palavras: incluir nas stopwords.
words_descarte = wordfreq_df['word'].iloc[5654+1:]
words_descarte = words_descarte.tolist()
stopwords_full = stopwords_pt + words_descarte

# Save:
pickle.dump(stopwords_full, open('/home/aphonsoamaral/files/stopwords_full.sav', 'wb'))
# Load:
stopwords_full = pickle.load(open('/home/aphonsoamaral/files/stopwords_full.sav', 'rb'))

#%%
# --------------------------------------------------------------------------------------------
# Bag of words com resultado COUNT:
# --------------------------------------------------------------------------------------------
# Time:
bow_count_start_time = time.time()

# Definir objeto da biblioteca, passando as stopwords.
vectorizer = CountVectorizer(stop_words=stopwords_full) # Passar a lista de stopwords
atendimento_vectors_count = vectorizer.fit_transform(atendimentos_sample) # Execução do algoritmo
# Save:
#pickle.dump(atendimento_vectors_count, open('/home/aphonsoamaral/files/atendimento_vectors_count_bag_of_words.sav', 'wb'))
# Load:
#atendimento_vectors_count = pickle.load(open('/home/aphonsoamaral/files/atendimento_vectors_count_bag_of_words.sav', 'rb'))

########################################################################################################################################
# A "Classificação_atendimentos.py" começa a partir daqui, do objeto "atendimento_vectors_count".
########################################################################################################################################

#%%
# Transformar em Bag of Words DataFrame
atendimento_vectors_count = atendimento_vectors_count.toarray() # Transforma em array, pois o algotitmo acima entrega em "csr_matrix"
atendimento_vectors_count = pd.DataFrame(atendimento_vectors_count) # Transforma o array em DataFrame
atendimento_vectors_count.columns = vectorizer.get_feature_names() # Passar as palavras como título do dataframe.

Bag_of_words_count = pd.merge(left=atendimentos_sample_df_bag_of_words[["Tipo_atendimento", 'id']],
                              right=atendimento_vectors_count,
                              how='inner',
                              left_index=True,
                              right_index=True)

#Bag_of_words_count.shape
# Exportar o Bag of Words COUNT
pd.DataFrame.to_csv(Bag_of_words_count, sep=';', path_or_buf='/home/aphonsoamaral/files/Bag_of_words_count.csv', index=False)

# Time:
bow_count_end_time = time.time()
Tempo_bow_count = round((bow_count_end_time - bow_count_start_time))
print('Tempo_bow_count: ' + str(Tempo_bow_count))

# # Buscar um registro do bag of words para ver as palavras dele
# teste1 = Bag_of_words_count.iloc[0]
# teste1 = pd.DataFrame(teste1)
# teste1 = teste1.reset_index()
# teste1.columns = ['palavra', 'freq']
# teste1 = teste1[teste1['freq'] != 0 ]

# Mostrar últimas colunas do dataframe
#Bag_of_words_count_aux.iloc[:,-5:].head()
#%%
# --------------------------------------------------------------------------------------------
# Tempos totais:
Tempos_execucao_BoW = {'Tempo_limpeza': Tempo_limpeza,
                       'Tempo_id_palavras': Tempo_id_palavras,
                       'Tempo_bow_count': Tempo_bow_count}

Tempos_execucao_BoW = pd.DataFrame(Tempos_execucao_BoW.items())
Tempos_execucao_BoW.columns = ['processo', 'segundos']
pd.DataFrame.to_csv(Tempos_execucao_BoW, sep=';', path_or_buf='/home/aphonsoamaral/files/Tempos_execucao_BoW.csv', index=False)