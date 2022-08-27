# -*- coding: utf-8 -*-
"""
@author: Julian Gil-Gonzalez,2022
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
from scipy import stats as st
from scipy.signal import butter,filtfilt
import neurokit2 as nk
import matplotlib.pyplot as plt

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
                
        self.corru[self.corru==0] = np.NaN        
        return self.corru
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)

    def resp_rate(self, x, corru, Fs):
        try: 
            signals, info = nk.rsp_process(x, sampling_rate=Fs,  method='biosppy')
            x_clean = signals['RSP_Clean'].to_numpy()
            ritmo_rsp = signals['RSP_Rate'].to_numpy()
            peaks = signals['RSP_Peaks'].to_numpy()
        except:
            x_clean = x
            ritmo_rsp = 20*np.ones(x.shape)
            peaks = np.zeros(x.shape)
        
        peaks[corru==1] = 0
                
        # Se identifica bradipnea. Resp rate < 12
        brad = np.nan*np.ones(x.shape)
        brad[ritmo_rsp < 12] = 1
        brad[corru==1] = np.NaN
        
        # Se identifica taquipnea. Resp rate > 20
        taq = np.nan*np.ones(x.shape)
        taq[ritmo_rsp > 20] = 1
        taq[corru==1] = np.NaN
        
        return x_clean, peaks, brad, taq
        