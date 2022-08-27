# -*- coding: utf-8 -*-
"""
@author: Julian Gil-Gonzalez,2022
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
from scipy import stats as st
from scipy.signal import butter,filtfilt

class anomalydetection_resp(BaseEstimator, TransformerMixin):
    
    """
        Detección de anomalías en señales de respiración. La señal, se divide en 
        ventanas de duración Tenvt segundos y se analiza su comportamiento estadístico
        con el fin de detectar si existen datos anómalos.

        Parámetros
        ----------
        Method : string
            Indica el método por el cual se quieren filtrar las series de
            tiempo. Tiene las siguientes opciones
                EEMD = Filtrado basado en la descomposición de modos empíricos.
                BandPass = se genera un filtro de Butterworth pasa-banda .
        x: list, numpy array
            La serie de tiempo a filtrar.
        Fs : int
            Frecuencia de muestreo de la señal a filtrar.
        Tenvt : int
            Duración de la ventana de análisis
            
                
        Salidas
        -------
        corru : numpy array, list
            una arreglo del mismo tamaño de la señal x, el cuál es uno si el dato es 
            considerado como anómalo.
    """
    
    def __init__(self, Fs=100, tenvt = 4):
        self.Fs = Fs # la tasa de muestreo.
        self.Win = int((tenvt)*Fs) # número de muestras en cada ventana
        
    def Search_MaxConsecutive(self, a):
      aux = 0
      Max = []
      Idx = []
      for i, ai in enumerate(a):
        if (i == len(a)-1) & (len(Max) == 0):
          Max.append(aux)
          Idx.append(i)
      
        if i == len(a)-1:
          i1 = i
        else:
          i1 = i - 1

        if ai == True:
          aux += 1
        elif (aux > 10) & (a[i1] == True):
          aux += 1
        elif aux != 0:
          Max.append(aux)
          aux = 0
          Idx.append(i-1)
      try:
        M = max(Max)
        idx = Idx[Max.index(M)]
      except:
        M = 0
        idx = -1
      return M, idx
        
    def fit(self, x, y=None):
        self.corru = np.zeros(x.shape)
        nyq = self.Fs
        cutoff = 50
        order = 5
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
            
    def transform(self, x, y=None):
        # Normalización de la señal
        aux = np.nanmax(x)
        if abs(aux)<1e-6:
            self.x = (x - np.nanmean(x))
        else:
            self.x = (x - np.nanmean(x))/np.nanmax(x)
        # Se recorren todas las ventanas en la señal x.
        for i in range(len(self.x)//self.Win):
            zt = self.x[i*self.Win:(i+1)*self.Win]
            # Se verifica la calidad de la señal
            # 1.la ventana se denomina corrupta si existe al menos un valor NaN (not a number).
            if np.sum(np.isnan(zt)):
              self.corru[i*self.Win:(i+1)*self.Win] = 1
            else: 
              # Si no hay datos NaN, se aplica un filtro pasa bajas con el fin de eliminar
              # componentes ruidos de alta frecuencia.
              zt = filtfilt(self.b, self.a, zt)
              
              # Se aplica un histograma de 10 bins a la señal con el fin de detectar zonas de 
              # saturación.
              hist, _ = np.histogram(zt)
              aux_ex = hist[0] + hist[-1]
              aux_mi = np.sum(hist[1:-1])
              
              # Una de las características particulares de las zonas saturadas, es la presencia de 
              # valores constantes en tiempos prolongados. Por esta razón se calcula la derivada de 
              # la señal y se verifica las zonas con valores cercanos a cero.
              grad = np.gradient(zt,1/self.Fs)
              aux_gr = abs(grad) <= 5e-2
              m, idx = self.Search_MaxConsecutive(aux_gr)
              
              if ((aux_ex - 0) > 0.5*self.Win) & (m/self.Fs > 0.3):
                self.corru[i*self.Win:(i+1)*self.Win] = 1
              if (idx != -1) & (m/self.Fs > 0.3) & ((abs(zt[idx] - zt.max()) < 0.05*zt.max()) | (abs(zt[idx] - zt.min()) < 0.05*zt.max())):
                self.corru[i*self.Win:(i+1)*self.Win] = 1
              elif ((aux_ex - 0) > 500) & (m/self.Fs > 0.3) & ((abs(st.mode(zt)[0][0]-x.max())<0.01*zt.max()) | ((abs(st.mode(zt)[0][0]-x.min())<0.01*zt.max()))):
                self.corru[i*self.Win:(i+1)*self.Win] = 1
              elif (aux_ex - aux_mi) > 0.95*self.Win:
                self.corru[i*self.Win:(i+1)*self.Win] = 1
              elif m/self.Fs > 0.5:
                self.corru[i*self.Win:(i+1)*self.Win] = 1
              elif zt.std() < 0.01:
                self.corru[i*self.Win:(i+1)*self.Win] = 1
                
        return self.corru
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)

   

class Normalizacion(BaseEstimator, TransformerMixin):
    
    """
        Normalización de series de tiempo.

        Parámetros
        ----------
        Method : string
            Indica el método por el cual se quieren normalizar las series de
            tiempo. 
            PicoDC: normalización a partir del valor DC de la señal y de se 
            valor máximo.
            Zscore: Normalización a partir de la media y desviación estandard. 
            Genera series de tiempo cuyos valores de amplitud tienen media 0 y 
            varianza unitaria.
        x: arreglo numpy
            Serie de tiempo
                
        Salidas
        -------
        x_nor: arreglo numpy 
            serie de tiempo normalizada
    """
    def __init__(self, Method='PicoDC', Plot=False, Fs = 0):
        self.Method = Method
        self.Plot = Plot
        self.Fs = Fs
        
    def fit_1(self, x, y = None):
        if self.Method == 'PicoDC':
            self.DC = np.mean(x) # se calcula el nivel DC de la señal
            self.max = np.max(np.abs(x)) # se calcula el valor máximo de la señal
        elif self.Method == 'Zscore':
            self.mean_ = np.mean(x)
            self.std_ = np.std(x)
            if self.std_ < 1e-9: # para favorecer la estabilidad numérica
                self.std_ = 1e-9
        else:
            raise Exception("El método de normalizacion no esta implementado")
        return self
    
    def fit(self, x, y = None):
        pass

    def transform(self, x):
        self.fit_1(x)
        if self.Method == 'PicoDC':
            if abs(self.max-self.DC) > 1e-3:
                x_nor = (x-self.DC)/(self.max-self.DC)
            else:
                x_nor = x
        elif self.Method == 'Zscore':
            x_nor = (x-self.mean_)/(self.std_)
        else:
            raise Exception("El método de normalizacion no esta implementado")

        return x_nor
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)
        
    

