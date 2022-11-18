# -*- coding: utf-8 -*-
"""
@author: Julian Gil-Gonzalez,2022
"""
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from AnomalyDetection_RESP import anomalydetection_resp
from AnomalyDetection_PPG import anomalydetection_ppg
from AnomalyDetection_ABP import anomalydetection_abp
from AnomalyDetection_ECG import anomalydetection_ecg
from MorphologicalCaracterizacion import caracterizacion
import numpy as np

class featureextraction(BaseEstimator, TransformerMixin):

  def __init__(self, Fs = 250, tvent = 0.8, train = 1):
    self.Fs    = Fs # Frecuencia de muestreo de las señales a parametrizar
    self.tvent = tvent # Ventana de análisis para encontrar zonas con datos 
                       # corruptos
    self.car   = caracterizacion(Fs = self.Fs) # Se carga el módulo para realizar
                                               # la caracterización.

  def anomaly(self, names):
    """
      Función para definir los objetos que realizan la búsqueda de zonas con 
      datos inválidos. Se usan 3 tipos de señales, a decir, PLETH, ABP y ECG.

      Parámetros:
      --------------------------------------
      names: list
        Los nombres de las señales que se van a procesar.
      Fs: int
        Frecuencia de muestreo de las señales.
      tvent: int/list/numpy array
        Tamaño de la ventana de análisis, si se aporta un único valor, se usará 
        el mismo tamaño de ventana para todas las señales de análisis.

      Salidas:
      --------------------------------------
      Anomaly: list
        Se genera una lista de objetos que ejecutarán las detecciones de datos
        inválidos
    """
    Anomaly = []
    try:
      Ns    = len(self.tvent)
      tvent = self.tvent
    except:
      tvent = [self.tvent]*len(names)
      
    for i, nam in enumerate(names): 
      if nam == 'PLETH': # Para señales de fotopletismografía
        a = anomalydetection_ppg(Fs = self.Fs, tenvt = tvent[i])
        Anomaly.append(a)

      if nam == 'ABP': # Para señales de presión arterial invasiva
        b = anomalydetection_abp(Fs = self.Fs, tenvt = tvent[i])
        Anomaly.append(b)

      if nam == 'ECG': # Para señales de electrocardiografía
        c = anomalydetection_ecg(Fs = self.Fs, tenvt = tvent[i])
        Anomaly.append(c)

    return Anomaly  

  def fit(self, X=None, y=None):
    pass

  
  def transform(self, X, y):
    """
      Función para realizar la caracterización de señales relacioandas con la
      actividad cardiaca

      Parámetros:
      --------------------------------------
      X: pandas DataFrame
        Un DataFrame pandas en donde cada columna obedece a un tipo de señal 
        (por ejemplo PLETH); similarmente, cada fila contiene las señales 
        capturadas para un sujeto específico. Si para un sujeto no se tiene 
        alguna de las señales, se debe representar con un NaN.
      y: list/numpy array
        Etiqueta de cada sujeto. 

      Salidas: X_t, Y_t, C_t
      --------------------------------------
      X_t: list
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene una matriz con las características extraídas
      Y_t: list
        Lista que contiene tantos elementos como tipos de señales capturadas.
        Cada elemento contiene las etiquetas de cada sujeto.
      C_t: list 
        Lista que contiene tantos elementos coomo tipos de señales capturadas.
        Cada elemento contiene una vector con la proporción de regiones con datos
        corruptos.
    """

    X['label'] = y
    names = X.columns[:-1] # tipo de señales a procesar

    X_t = []
    Y_t = []
    C_t = []
    for i, nam in enumerate(names):
      # Se definen los objetos para el procesamiento y la caracterización 
      Anomaly = self.anomaly(names) 
      car = self.car

      sig_  = X[[nam, 'label']]
      x_sig = sig_[nam]
      y_sig = sig_['label'].to_numpy()
      
      aux  = 0
      X_   = np.zeros((len(x_sig), 14))
      Y_   = -1*np.ones(len(x_sig))
      Corr = -1*np.ones(len(x_sig))
      for j, x in enumerate(x_sig):
        if isinstance(x, list):
          x = x[0]
        elif  isinstance(x, float):
          x = np.NaN*np.ones(200)
        if len(x.shape) > 1:
          x = x.flatten() 

        # Se detectan las zonas inválidas de la señal
        corru_p = Anomaly[i].fit_transform(x)
        Corr[aux] = corru_p.sum()/len(x)

        # Solo se caracterizan únicamente las señales donde la longitud de las
        # zonas inválidas no superan el 70% de la totalidad de la señal. 
        Y_[aux]    = y_sig[aux]
        if corru_p.sum() < 0.7*len(x): 
          X_[aux, :] = car.fit_transform(x)
        else:
          X_[aux, :] = np.zeros((1, 14))

        aux += 1

      C_t.append(Corr)
      X_t.append(X_)
      Y_t.append(Y_)
    return X_t, Y_t, C_t

  def fit_transform(self, X, y):
    return self.transform(X,y)

      

