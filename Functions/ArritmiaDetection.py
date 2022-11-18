# -*- coding: utf-8 -*-
"""
@author: Julian Gil-Gonzalez,2022
"""
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ArritmiaDetection(BaseEstimator, ClassifierMixin):

  def __init__(self):
    pass

  def fit(self, X, y, C):
    """
      Función entrenar el modelo de detección de dos arritmias, a decir, 
      bradicardia y taquicardia. El modelo se construye a partir del ensamble de 
      tantos clasificadores como señales. Por ejemplo, si se tienen dos tipos de
      señales (por ejemplo, ABP y PLETH), entonces es necesario entrenar 2 
      algoritmos de clasificación.

      Parámetros:
      --------------------------------------
      X: list
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene una matriz con las características extraídas. 
        Dichas caracerísticas son usadas para el entrenamiento de los 
        clasificadores
      y: list
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene las etiquetas de cada sujeto.
      C: list 
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene una vector con la proporción de regiones con datos
        corruptos.

      Salidas:
      --------------------------------------
      retorna un objeto con los clasificadores entrenados 
    """
    
    if isinstance(X, list):
      N_s = len(X)
    else:
      N_s = 1
    self.Cl = []
    for j in range(N_s):
      X_ = X[j]
      y_ = y[j]
      C_ = C[j]

      # Para entrenar los clasificadores solo se eligen las muestras 
      # donde la longitud de las zonas inválidas no supere el 70% de la 
      # totalidad de la señal.
      X_ = X_[C_<0.7,:]
      y_ = y_[C_<0.7]
      self.Cl.append(Pipeline([("scaler", StandardScaler()),
                          ("RF", RandomForestClassifier(n_estimators = 500,\
                                                        min_samples_leaf=6,\
                                                        random_state=42))]).fit(X_, y_))
    return self.Cl
  
  def predict(self, X, C):
    """
      Función realizar la detección de bradicardia y taquicardia.

      Parámetros:
      --------------------------------------
      X: list
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene una matriz con las características extraídas. 
        Dichas caracerísticas son usadas para la detección de bradicardia y 
        taquicardia a partir de los clasificadores previamente entrenados.
      C: list 
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene una vector con la proporción de regiones con datos
        corruptos.

      Salidas:
      --------------------------------------
      prediction: list/numpy array
        la predicción de la anomalía detectada. Dicha predicción se hace a partir 
        del ensamble de múltiples clasificadores (uno por tipo de señal) ponderado 
        por un factor relacionado con la razón de zonas con datos corruptos de 
        cada señal 
        Ejemplo: Considerar que se para un sujeto particular se tiene dos tipos
        de señales: ABP y PLETH. En este caso se tienen dos clasificadores, uno 
        por cada señal. Suponer que con base en las características extraídas de 
        cada señal se tienen dos predicciones, una por cada señal, a decir, 
        p_PLETH p_ABP. Cada predicción tiene asociado un peso, w_PLETH = 1 - C_PLET,
        donde C_PLET es la proporción de zonas inválidad de la señal (para la señal 
        ABP, se tiene w_ABP = 1 - C_ABP). En este sentido la predicción para el 
        sujeto en particual se calcula como:

        pred = round((1/(w_ABP + w_PLETH))*(w_ABP*p_ABP + p_PLETH*w_PLETH)),
        donde round es la función que aproxima al entero más cercano.
    """

    if isinstance(X, list):
      N_s = len(X)
      N   = X[0].shape[0]
    else:
      N_s = 1
    
    pred_ = np.zeros((N, N_s))
    self.w_    = np.zeros((N, N_s))
    # Predicción para todas las señales en la base de datos.
    for j in range(N_s):
      X_ = X[j]
      C_ = C[j]
      
      # Se efectúa la predicción únicamente para las señales válidas.
      idx_  = np.where(C_ < 0.7)[0]
      X_aux = X_[idx_, :]
      pred_[idx_, j] = self.Cl[j].predict(X_aux)
      self.w_[:, j] = 1 - C_ # los pesos para la ponderación. Nótese que es el 
                             # complemento de la proporción de regiones con datos
                             # anómalos; así que una señal con menos regiones corruptas
                             # tiene mayor aporte en la predicción.
                           

    norm = np.sum(self.w_,axis=1)
    norm[norm == 0] = 1
    self.prediction = np.round(np.diag(self.w_@pred_.T)/norm)
    return self.prediction