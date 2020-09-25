import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV # Exemplo Github nos classificadores imbutidos nos pipelines: https://github.com/scikit-learn/scikit-learn/issues/12728
#import scipy.stats as ss
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time

#%%
# Carregar arquivos:

# Base full de atendimentos - origem: "2.3.2.Bag_of_words_atendimentos.py"
atendimentos = pickle.load(open('/home/aphonsoamaral/files/atendimentos_df_full.sav', 'rb'))

# Lista com a descrição dos atendimentos já tratada - origem: "2.3.2.Bag_of_words_atendimentos.py":
atendimentos_sample = pickle.load(open('/home/aphonsoamaral/files/atendimentos_sample_list.sav', 'rb'))

# Carregar dataframe da amostra - origem: "2.4.1.Dataset_atendimentos_sample.py":
sample1 = pickle.load(open('/home/aphonsoamaral/files/sample1_df.sav', 'rb'))

# Carregar amostra classificada:
# Copiar arquivos p/ servidor:
# scp /home/aphonsoamaral/aphonso/DSBD-UFPR/TCC/Comp_cliente/z_Scripts_apoio/sample1_atendimentos_classificada.csv aphonsoamaral@35.197.108.154:/home/aphonsoamaral/files

# Carregar amostra classificada:
sample1_class = pd.read_csv('/home/aphonsoamaral/files/sample1_atendimentos_classificada.csv',
                            sep='|',
                            dtype = {'Tipo_atendimento': str,
                                     'id': str,
                                     'Frente_atendimento': str,
                                     'Classificacao': str})

# Carregar stopwords criadas no arquivo "2.3.2.Bag_of_words_atendimentos.py"
# Load:
stopwords_full = pickle.load(open('/home/aphonsoamaral/files/stopwords_full.sav', 'rb'))

#%%
# Tratar arquivos:
    # 1. Substituir o texto do objeto "atendimentos" pelo texto do "atendimentos_sample" (que já está tratado)
    # 2. Filtrar no dataframe do item 1, somente os itens da amostra
    # 3. Gerar lista apenas dos itens da amostra, a partir da coluna "texto", gerado pelo DataFrame filtrado do item 2

# Item 1:
atendimentos_sample_df = pd.DataFrame(atendimentos_sample)
atendimentos_sample_df.columns = ['Texto_tratado']
atendimentos_trat_1 = pd.merge(left=atendimentos[["Tipo_atendimento", 'id', 'organization_cnpj', 'Frente_atendimento']],
                               right=atendimentos_sample_df['Texto_tratado'],
                               how='inner',
                               left_index=True,
                               right_index=True)
# Item 2:
atendimentos_trat_2 = pd.merge(left=atendimentos_trat_1[["Tipo_atendimento", 'id', 'organization_cnpj', 'Frente_atendimento', 'Texto_tratado']],
                               right=sample1[['Tipo_atendimento', 'id']],
                               how='inner',
                               left_on=['Tipo_atendimento', 'id'],
                               right_on=['Tipo_atendimento', 'id'])
# Item 3:
atendimentos_trat_3 = pd.merge(left=atendimentos_trat_2[["Tipo_atendimento", 'id', 'organization_cnpj', 'Frente_atendimento', 'Texto_tratado']],
                               right=sample1_class[['Tipo_atendimento', 'id', 'Classificacao']],
                               how='inner',
                               left_on=['Tipo_atendimento', 'id'],
                               right_on=['Tipo_atendimento', 'id'])


#%%
# Tratar dados para ML:
X = atendimentos_trat_3['Texto_tratado'].to_list()
Y = atendimentos_trat_3["Classificacao"].to_numpy()

# Passar o OneHotEncoding na classificação (dúvida, problema, solicitação)
class1 = LabelEncoder()
atendimentos_trat_3['Class'] = class1.fit_transform(atendimentos_trat_3["Classificacao"])

# Random state (seed)
rand_st = 3

# Separar dados entre treino e teste:
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3, # Tamanho % da base de teste
                                                    random_state=rand_st)

#%%
# --------------------------------------------------------------------------------------------------------------------------------------------
# Algoritmo 1: Naive Bayes Classifier (NB)
# --------------------------------------------------------------------------------------------------------------------------------------------
# Criar pipeline e treinar modelo:
pip_NB = Pipeline([('vect', CountVectorizer(stop_words=stopwords_full)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())])
pip_NB = pip_NB.fit(X_train, y_train)

# Ver parâmetros do algoritmo:
#MultinomialNB().get_params().keys()

#%%
# Avaliar acurácia na base de teste [default]:
predicted_NB = pip_NB.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy
print('Accuracy base de treino -> NB: ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_NB.predict(X_train)) *100, 2) )+'%')

# Test accuracy
print('Accuracy base de teste -> NB: ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_NB) *100, 2) )+'%')

#%%
# GridSearch para NB Classifier:
# Parâmetros do GridSearch:
parameters_NB = {#'vect__max_df': (1e-1, 9e-1),
                 #'vect__min_df': (1e-1, 9e-1),
                 'vect__binary': (True, False),
                 'tfidf__norm': ('l1', 'l2'),
                 'tfidf__use_idf': (True, False),
                 'tfidf__smooth_idf': (True, False),
                 'clf__alpha': (1e-1, 1e-10), # alpha é um hiperparâmetro > https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)
                 'clf__fit_prior': (True, False),
                 }

#%%
# Rodar GridSearch:
# O GridSearchCV() também faz validação cruzada.
gs_NB = GridSearchCV(estimator=pip_NB, param_grid=parameters_NB, cv=5, n_jobs=-1)
gs_NB = gs_NB.fit(X_train, y_train)

# Lastly, to see the best mean score and the params, run the following code:
print('Score com melhor combinação de parâmetros -> NB: ' + str( round(gs_NB.best_score_ *100,2) ) +'%')
print('Melhor combinação de parâmetros -> NB: ' + str(gs_NB.best_params_))

# 0.3: Melhor combinação de parâmetros: {'clf__alpha': 0.1, 'clf__fit_prior': False, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__use_idf': True, 'vect__binary': False}
# Apenas "fit_prior" alterado. Com 'True' o algoritmo identifica a distribuiçao de probabiliade das palavras antes de treinar. O false considera distribuiçao normal.
#%%
# Treinar o modelo com o melhor estimador (melhor combinação de parâmetros encontrada pelo GridSearch)
pip_NB_best = gs_NB.best_estimator_.fit(X_train, y_train)

# Avaliar acurácia na base de teste [best]:
predicted_NB_best = pip_NB_best.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy best param:
print('Accuracy base de treino -> NB (best params): ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_NB_best.predict(X_train)) *100, 2) )+'%')

# Test accuracy best param:
print('Accuracy base de teste -> NB (best params): ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_NB_best) *100, 2) )+'%')


#%%
# Validação cruzada com 5 folds
scores_NB_best = cross_val_score(pip_NB_best, X_train, y_train, cv=5, scoring='accuracy')
print('Scores da validação cruzada -> NB (best params): ' +str(scores_NB_best))
print('Média dos scores da validação cruzada -> NB (best params): ' + str( round(scores_NB_best.mean() * 100, 2) ) +'%')

#%%

# Classification report
# classification_report(y_test,gs_NB.best_estimator_.predict_proba(X_test))
# ND_pred = best_svc.predicted_NB_best
# print(classification_report(y_test,predicted_NB_best))

#%%
# Matriz de confusão do NB Classifier
class_names = list(dict.fromkeys(Y.tolist()))
# Plot non-normalized confusion matrix
titles_options = [("NB - Confusion matrix, without normalization", None),
                  ("NB - Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(pip_NB_best, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.tight_layout()
    plt.show()

# Salvar modelo NB:
#pickle.dump(pip_NB_best, open('/home/aphonsoamaral/files/pip_NB_best.sav', 'wb'))
# Load:
#pip_NB_best = pickle.load(open('/home/aphonsoamaral/files/pip_NB_best.sav', 'rb'))


#%%
# --------------------------------------------------------------------------------------------------------------------------------------------
# Algoritmo 2: Stochastic Gradient Descent Classifier (SGD): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
# --------------------------------------------------------------------------------------------------------------------------------------------
pip_SGD = Pipeline([('vect', CountVectorizer(stop_words=stopwords_full)),
                    ('tfidf', TfidfTransformer()),
                    ('clf-sgd', SGDClassifier(max_iter=1000, random_state=rand_st))]) # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
pip_SGD = pip_SGD.fit(X_train, y_train)

# Ver parâmetros do algoritmo:
#SGDClassifier().get_params().keys()

#%%
# Avaliar performance [default]:
predicted_SGD = pip_SGD.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy
print('Accuracy base de treino -> SGD: ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_SGD.predict(X_train)) *100, 2) )+'%')

# Test accuracy
print('Accuracy base de teste -> SGD: ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_SGD) *100, 2) )+'%')

#%%
# GridSearch para SGD

# Time:
GS_SGD_start = time.time()

parameters_SGD = {'vect__binary': (True, False),
                  'tfidf__norm': ('l1', 'l2'),
                  'tfidf__use_idf': (True, False),
                  'tfidf__smooth_idf': (True, False),
                  'clf-sgd__alpha': (1e-1, 1e-10),
                  'clf-sgd__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
                  'clf-sgd__penalty': ('l1', 'l2'),
                  'clf-sgd__fit_intercept': (True, False),
                  'clf-sgd__shuffle': (True, False),
                  'clf-sgd__early_stopping': (True, False)
                 }

#%%
# Rodar GridSearch:
gs_SGD = GridSearchCV(estimator=pip_SGD, param_grid=parameters_SGD, cv=5, n_jobs=-1)
gs_SGD = gs_SGD.fit(X_train, y_train)
# [Mensagem]: /home/aphonsoamaral/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_stochastic_gradient.py:570: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit. warnings.warn("Maximum number of iteration reached before "

print('Score com melhor combinação de parâmetros -> SGD (best params): ' + str( round(gs_SGD.best_score_ *100,2) ) +'%')
print('Melhor combinação de parâmetros -> SGD (best params): ' + str(gs_SGD.best_params_))
# {'clf-sgd__alpha': 1e-10, 'clf-sgd__early_stopping': False, 'clf-sgd__fit_intercept': True, 'clf-sgd__loss': 'log', 'clf-sgd__penalty': 'l1', 'clf-sgd__shuffle': False, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__use_idf': False, 'vect__binary': True}

# Time:
GS_SGD_end = time.time()
Tempo_GS_SGD = round((GS_SGD_end - GS_SGD_start))
print('Tempo_GS_SGD: ' + str(Tempo_GS_SGD)) # Tempo_GS_SGD: 4704 (78 min)

#%%

# Treinar o modelo com o melhor estimador (melhor combinação de parâmetros encontrada pelo GridSearch)
pip_SGD_best = gs_SGD.best_estimator_.fit(X_train, y_train)

# Avaliar acurácia na base de teste [best]:
predicted_SGD_best = pip_SGD_best.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy best param:
print('Accuracy base de treino -> SGD (best params): ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_SGD_best.predict(X_train)) *100, 2) )+'%')

# Test accuracy best param:
print('Accuracy base de teste -> SGD (best params): ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_SGD_best) *100, 2) )+'%')

#%%
# Validação cruzada com 5 folds

# Time:
CV_SGD_start = time.time()

scores_SGD_best = cross_val_score(pip_SGD_best, X_train, y_train, cv=5, scoring='accuracy')
print('Scores da validação cruzada -> SGD (best params): ' +str(scores_SGD_best))
print('Média dos scores da validação cruzada -> SGD (best params): ' + str( round(scores_SGD_best.mean() * 100, 2) ) +'%')

# Time:
CV_SGD_end = time.time()
Tempo_CV_SGD = round((CV_SGD_end - CV_SGD_start))
print('Tempo_CV_SGD: ' + str(Tempo_CV_SGD))


#%%
# Matriz de confusão do SGD
class_names = list(dict.fromkeys(Y.tolist()))
# Plot non-normalized confusion matrix
titles_options = [("SGD - Confusion matrix, without normalization", None),
                  ("SGD - Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(pip_SGD_best, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Greens,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.tight_layout()
    plt.show()

# Salvar modelo SGD:
#pickle.dump(pip_SGD_best, open('/home/aphonsoamaral/files/pip_SGD_best.sav', 'wb'))
# Load:
#pip_SGD_best = pickle.load(open('/home/aphonsoamaral/files/pip_SGD_best.sav', 'rb'))

#%%
# --------------------------------------------------------------------------------------------------------------------------------------------
# Algoritmo 3: C-Support Vector Classification (SVC): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# --------------------------------------------------------------------------------------------------------------------------------------------
# Criar pipeline e treinar modelo:
pip_SVC = Pipeline([('vect', CountVectorizer(stop_words=stopwords_full)),
                    ('tfidf', TfidfTransformer()),
                    ('clf-svc', svm.SVC(random_state=rand_st))])
pip_SVC = pip_SVC.fit(X_train, y_train)

# Ver parâmetros do algoritmo:
#svm.SVC().get_params().keys()

#%%
# Avaliar acurácia na base de teste [default]:
predicted_SVC = pip_SVC.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy
print('Accuracy base de treino -> SVC: ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_SVC.predict(X_train)) *100, 2) )+'%')

# Test accuracy
print('Accuracy base de teste -> SVC: ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_SVC) *100, 2) )+'%')

#%%
# GridSearch:
parameters_SVC = {'vect__binary': (True, False),
                  'tfidf__norm': ('l1', 'l2'),
                  'tfidf__use_idf': (True, False),
                  'tfidf__smooth_idf': (True, False),
                  'clf-svc__C': [.0001, .001, .01],
                  'clf-svc__gamma': [.0001, .001, .01, .1, 1, 10, 100],
                  'clf-svc__degree': [1, 2, 3, 4, 5],
                  'clf-svc__kernel': ['linear', 'rbf', 'poly'],
                  }

#%%
# Rodar GridSearch:
# Time:
GS_SVC_start = time.time()

gs_SVC = GridSearchCV(estimator=pip_SVC, param_grid=parameters_SVC, cv=5, n_jobs=-1)
gs_SVC = gs_SVC.fit(X_train, y_train)

print('Score com melhor combinação de parâmetros -> SVC: ' + str( round(gs_SVC.best_score_ *100,2) ) +'%')
print('Melhor combinação de parâmetros -> SVC: ' + str(gs_SVC.best_params_))

# Melhor combinação de parâmetros -> SVC: {'clf-svc__C': 0.01, 'clf-svc__degree': 1, 'clf-svc__gamma': 100, 'clf-svc__kernel': 'poly', 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__use_idf': True, 'vect__binary': True}

#%%
# Treinar o modelo com o melhor estimador (melhor combinação de parâmetros encontrada pelo GridSearch)
pip_SVC_best = gs_SVC.best_estimator_.fit(X_train, y_train)

# Avaliar acurácia na base de teste [best]:
predicted_SVC_best = pip_SVC_best.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy best param:
print('Accuracy base de treino -> SVC (best params): ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_SVC_best.predict(X_train)) *100, 2) )+'%')

# Test accuracy best param:
print('Accuracy base de teste -> SVC (best params): ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_SVC_best) *100, 2) )+'%')

# Time:
GS_SVC_end = time.time()
Tempo_GS_SVC = round((GS_SVC_end - GS_SVC_start))
print('Tempo_GS_SVC: ' + str(Tempo_GS_SVC)) # Tempo_GS_SVC: 5157 (86 min)

#%%
# Validação cruzada com 5 folds
scores_SVC_best = cross_val_score(pip_SVC_best, X_train, y_train, cv=5, scoring='accuracy')
print('Scores da validação cruzada -> SVC (best params): ' +str(scores_SVC_best))
print('Média dos scores da validação cruzada -> SVC (best params): ' + str( round(scores_SVC_best.mean() * 100, 2) ) +'%')

#%%
# Matriz de confusão do SVC
class_names = list(dict.fromkeys(Y.tolist()))
# Plot non-normalized confusion matrix
titles_options = [("SVC - Confusion matrix, without normalization", None),
                  ("SVC - Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(pip_SVC_best, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Reds,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.tight_layout()
    plt.show()

# Salvar modelo SVC:
#pickle.dump(pip_SVC_best, open('/home/aphonsoamaral/files/pip_SVC_best.sav', 'wb'))
# Load:
#pip_SVC_best = pickle.load(open('/home/aphonsoamaral/files/pip_SVC_best.sav', 'rb'))


#%%
# --------------------------------------------------------------------------------------------------------------------------------------------
# Algoritmo 4: Random Forest Classifier (RFC): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# --------------------------------------------------------------------------------------------------------------------------------------------
# Criar pipeline e treinar modelo:
pip_RFC = Pipeline([('vect', CountVectorizer(stop_words=stopwords_full)),
                    ('tfidf', TfidfTransformer()),
                    ('clf-rfc', RandomForestClassifier(random_state=rand_st))])

pip_RFC = pip_RFC.fit(X=X_train, y=y_train)


#%%
# Avaliar acurácia na base de teste [default]:
predicted_RFC = pip_RFC.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy
print('Accuracy base de treino -> RFC: ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_RFC.predict(X_train)) *100, 2) )+'%')

# Test accuracy
print('Accuracy base de teste -> RFC: ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_RFC) *100, 2) )+'%')

#%%
# Randomized Search Cross Validation
# n_estimators
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
RFC_random_grid = {'vect__binary': (True, False),
                   'tfidf__norm': ('l1', 'l2'),
                   'tfidf__use_idf': (True, False),
                   'tfidf__smooth_idf': (True, False),
                   'clf-rfc__n_estimators': n_estimators,
                   'clf-rfc__max_features': max_features,
                   'clf-rfc__max_depth': max_depth,
                   'clf-rfc__min_samples_split': min_samples_split,
                   'clf-rfc__min_samples_leaf': min_samples_leaf,
                   'clf-rfc__bootstrap': bootstrap}

# Definition of the random search
rs_RFC = RandomizedSearchCV(estimator=pip_RFC,
                            param_distributions=RFC_random_grid,
                            n_iter=50,
                            scoring='accuracy',
                            cv=5,
                            verbose=1,
                            random_state=rand_st,
                            n_jobs=-1)

# Fit the random search model
rs_RFC = rs_RFC.fit(X=X_train, y=y_train)
#%%

print('Score com melhor combinação de parâmetros -> RFC: ' + str( round(rs_RFC.best_score_ *100,2) ) +'%')
print('Melhor combinação de parâmetros -> RFC: ' + str(rs_RFC.best_params_))
# Melhor combinação de parâmetros -> RFC: # {'vect__binary': False, 'tfidf__use_idf': True, 'tfidf__smooth_idf': False, 'tfidf__norm': 'l2', 'clf-rfc__n_estimators': 800, 'clf-rfc__min_samples_split': 10, 'clf-rfc__min_samples_leaf': 2, 'clf-rfc__max_features': 'sqrt', 'clf-rfc__max_depth': 40, 'clf-rfc__bootstrap': False}

#%%
# Treinar o modelo com o melhor estimador (melhor combinação de parâmetros encontrada pelo RandomSearch)
pip_RFC_best = rs_RFC.best_estimator_.fit(X_train, y_train)

# Avaliar acurácia na base de teste [best]:
predicted_RFC_best = pip_RFC_best.predict(X_test) # É aqui que os dados são rotulados

# Training accuracy best param:
print('Accuracy base de treino -> RFC (best params): ' + str( round(accuracy_score(y_true=y_train, y_pred=pip_RFC_best.predict(X_train)) *100, 2) )+'%')

# Test accuracy best param:
print('Accuracy base de teste -> RFC (best params): ' + str( round(accuracy_score(y_true=y_test, y_pred=predicted_RFC_best) *100, 2) )+'%')

#%%
# Validação cruzada com 5 folds
scores_RFC_best = cross_val_score(pip_RFC_best, X_train, y_train, cv=5, scoring='accuracy')
print('Scores da validação cruzada -> RFC (best params): ' +str(scores_RFC_best))
print('Média dos scores da validação cruzada -> RFC (best params): ' + str( round(scores_RFC_best.mean() * 100, 2) ) +'%')

#%%
# Matriz de confusão do RFC
class_names = list(dict.fromkeys(Y.tolist()))
# Plot non-normalized confusion matrix
titles_options = [("RFC - Confusion matrix, without normalization", None),
                  ("RFC - Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(pip_RFC_best, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Oranges,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.tight_layout()
    plt.show()

# Salvar modelo RFC:
#pickle.dump(pip_RFC_best, open('/home/aphonsoamaral/files/pip_RFC_best.sav', 'wb'))
# Load:
#pip_RFC_best = pickle.load(open('/home/aphonsoamaral/files/pip_RFC_best.sav', 'rb'))

#%%
# --------------------------------------------------------------------------------------------------------------------------------------------
# Classificação da base de atendimentos:
# --------------------------------------------------------------------------------------------------------------------------------------------
# Classificar o texto conforme modelo Naive Bayes:
atendimentos_trat_1["Classificacao_NB"] = pip_NB_best.predict(atendimentos_trat_1["Texto_tratado"])

# Salvar base classificada com o melhor modelo:
#pickle.dump(atendimentos_trat_1, open('/home/aphonsoamaral/files/atendimentos_classificados_NB_Classifier.sav', 'wb'))
# Load:
#atendimentos_classificados_NB_Classifier = pickle.load(open('/home/aphonsoamaral/files/atendimentos_classificados_NB_Classifier.sav', 'rb'))
