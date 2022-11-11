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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
from scipy import stats as st
from scipy.signal import butter,filtfilt

class anomalydetection_ppg(BaseEstimator, TransformerMixin):
    
    """
        Detección de zonas inválidas en señales de fotopletismografía. La señal, 
        se divide en ventanas de duración "Tenvt" segundos y se analiza su 
        comportamiento estadístico con el fin de detectar si existen datos 
        inválidos.

        Parámetros
        ----------
        x: list, numpy array
            La serie de tiempo que se quiere procesar.
        Fs : int
            Frecuencia de muestreo de la señal.
        Tenvt : int
            Duración de la ventana de análisis en segundos
            
                
        Salidas
        -------
        corru : numpy array, list
            una arreglo del mismo tamaño de la señal x, el cuál es uno si el dato es 
            considerado como inválido.
    """
    
    def __init__(self, Fs=100, tenvt = 0.8):
        self.Fs = Fs # la tasa de muestreo.
        self.Win = int((tenvt)*Fs) # número de muestras en cada ventana
        
    def fit(self, x, y=None):
        self.corru = np.zeros(x.shape)
        self.corru_a = np.zeros(x.shape)
        nyq = self.Fs
        cutoff = 1
        order = 2
        normal_cutoff = cutoff / nyq
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)
            
    def transform(self, x, y=None):
        self.x = x
        
        
        if x.std() < 0.01:
          self.corru = np.ones(self.corru.shape)
          #print(7)
        else:
          # Se recorren todas las ventanas en la señal x.
          aux_corrup = 0 
          for i in range(len(self.x)//self.Win):
              
              zt = self.x[i*self.Win:(i+1)*self.Win]
              
              # Se verifica la calidad de la señal

              # 1.la ventana se denomina corrupta si existe al menos un valor 
              # NaN (not a number).
              if np.sum(np.isnan(zt)):
                self.corru[i*self.Win:(i+1)*self.Win] = 1
                #print(1)
              else: 

                if zt.std() <= 0.0001:
                  self.corru[i*self.Win:(i+1)*self.Win] = 1
                  #print(3)
                else: 
                  # Se calculan las pendientes de los datos, para detectar la anomalía 
                  # donde la amplitud empieza a oscilar arbitraramente
                  slope = np.zeros(zt.shape)
                  for n in range(0,len(zt)):
                    if n == 0:
                      slope[n] = np.sign(zt[n])
                    else:
                      slope[n] = np.sign(zt[n]-zt[n-1])
                  aux_cons = 0
                  aux_riza = 0
                  
                  for n, sl in enumerate(slope):
                    if sl == 0:
                      aux_cons += 1
                    else:
                      aux_cons = 0
                      if (n != 0):
                        if slope[n-1]*slope[n] == -1:
                          aux_riza += 1

                    if (aux_cons > 0.5*self.Win) | (aux_riza > 0.5*self.Win):
                      self.corru[i*self.Win:(i+1)*self.Win] = 1
                      #print(4)
                      break

              # Se calcula el histograma de la señal con el fin de identificar 
              # regiones donde la señal se satura.
              if np.sum(self.corru[i*self.Win:(i+1)*self.Win]) == 0:
                hist, _ = np.histogram(zt)
                
                idx = np.argsort(hist)
                if (hist[0]>0.25*self.Win) | (hist[1]>0.25*self.Win):
                  if (hist[8]>0.25*self.Win) | (hist[9]>0.25*self.Win):
                    self.corru[i*self.Win:(i+1)*self.Win] = 1
                    #print(5)
              
        return self.corru
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)
        
    def heart_rate(self, x, corru, Fs):
        try: 
            signals, info = nk.ppg_process(x, sampling_rate=Fs)
            x_clean = signals['PPG_Clean'].to_numpy()
            peaks = signals['PPG_Peaks'].to_numpy()
        except:
            x_clean = nk.ppg_clean(x)
            peaks_ = processing.find_local_peaks(x, 10)
            peaks = np.zeros(x.shape)
            if len(peaks_) !=0:
              peaks[peaks_] = 1
              
        corrected_peak_inds = np.where(peaks == 1)[0]
        ritmo_ppg = processing.compute_hr(len(x), corrected_peak_inds, Fs)


        peaks[corru==1] = 0
        t_peak = corrected_peak_inds
        x_peak = x[corrected_peak_inds]
        
        
                
        # Se identifica bradipnea. Resp rate < 12
        brad = np.nan*np.ones(x.shape)
        brad[ritmo_ppg < 40] = 1
        brad[corru==1] = np.NaN
        
        # Se identifica taquipnea. Resp rate > 20
        taq = np.nan*np.ones(x.shape)
        taq[ritmo_ppg > 140] = 1
        taq[corru==1] = np.NaN
        
        return x_clean, peaks, brad, taq

        