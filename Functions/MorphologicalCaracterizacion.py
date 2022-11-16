# -*- coding: utf-8 -*-
"""
@author: Julian Gil-Gonzalez,2022
"""
import scipy.io as sio
import numpy as np
import neurokit2 as nk
from scipy.stats import norm, kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin


class caracterizacion_1(BaseEstimator, TransformerMixin):
    
    """
        Caracterización de series de tiempo a partir de descriptores fisiológicos

        Parámetros
        ----------
        x: list, numpy array
            la serie de tiempo a ser caracterizada.
        Fs : int
            Frecuencia de muestreo de la señal a filtrar.
            
                
        Salidas
        -------
        X: numpy array
            Vector de características.
    """
    
    def __init__(self, Fs=100):
        self.Fs = Fs # tasa de muestreo
        
    def fit(self, x):
        return self
        
    def transform(self, x):
        signals, _ = nk.ppg_process(x, sampling_rate=self.Fs)
        x = signals['PPG_Clean']
        HR, _ = nk.ppg_process(x, sampling_rate=self.Fs)
        HR = HR['PPG_Rate'].to_numpy()
        peaks = nk.ppg_findpeaks(x, self.Fs)["PPG_Peaks"]
        feat_1 = nk.hrv_time(peaks, sampling_rate=self.Fs, show=False)
        Xn     = np.zeros((1,14))
        Xn[:,0]  = skew(x)
        Xn[:,1]  = kurtosis(x)
        Xn[:,2]  = feat_1['HRV_MaxNN'].to_numpy()[0]/1000
        Xn[:,3]  = np.min(self.ComputeHRBeats(peaks, 4))
        Xn[:,4]  = np.max(self.ComputeConBeat(HR, peaks, 46, 0))
        Xn[:,5]  = np.max(self.ComputeHRBeats(peaks, 12))
        Xn[:,6]  = np.max(self.ComputeConBeat(HR, peaks, 130, 1))
        Xn[:,7]  = np.max(HR)
        Xn[:,8]  = feat_1['HRV_SDNN'].to_numpy()[0]/1000
        Xn[:,9]  = np.std(x)
        Xn[:,10] = len(peaks)
        Xn[:,11] = feat_1['HRV_MedianNN'].to_numpy()[0]/1000
        Xn[:,12] = np.max(self.ComputeHRBeats(peaks, 4))
        Xn[:,13] = np.max(self.ComputeConBeat(HR, peaks, 95, 1))
        return Xn
    
    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)

    def ComputeHRBeats(self, peaks, B):
        #Calcula el ritmo cardiaco medio en una ventana de B beats
        #
        #Parámetros
        #----------
        #HR: list, numpy array
        #    serie de tiempo que representa el ritmo cardiaco instantaneo
        #peaks: list, numpy array
        #    arreglo que contiene la ubicación de los picos.
        #B: int
        #    tamaño de la ventana de análisis dado en número de Beats.
        #
        #
        #Salidas
        #-------
        #HR: list, numpy array
        #    arreglo que contiene los ritmos cardiacos en cada ventana de análisis


        idx1 = 0
        idx2 = idx1 + B - 1
        N   = len(peaks)
        HR  = []
        while idx2 < N:
            delta = (peaks[idx2] - peaks[idx1])/self.Fs
            HR.append(60*B/delta)
            idx1 += 1
            idx2 = idx1 + B
        if len(HR) == 0:
          HR.append(0)
        return np.array(HR)

    def ComputeConBeat(self, HR, peaks, um, mayor):

        #Calcula el número de beats consecutivos que ocurren mientras el ritmo cardiaco se encuentra en un intervalo
        #particulas. Por ejmeplo, si um=40 y mayor = 0, se calcula el número de beats ocurridos consecutivamente mientras
        #el ritmo cardiaco es menor a 40 beats por minuto.
        #
        #Parámetros
        #----------
        #HR: list, numpy array
        #    serie de tiempo que representa el ritmo cardiaco instantaneo
        #peaks: list, numpy array
        #    arreglo que contiene la ubicación de los picos.
        #um: int
        #    Umbral para el ritmo cardiaco.
        #mayor: Binary
        #    Define si el umbral es mayor que o menor que.
        #
        #Salidas
        #-------
        #Npeaks: list, numpy array
        #    arreglo que contiene la cantidad de picos consecutivos ocurridos en el umbral definido.

        if mayor == 1:
            aux_cond = HR > um
        else:
            aux_cond = HR < um
        
        a = 0
        idx_p = -1
        Auxidx = []
        N = len(aux_cond)
        while a == 0:
          aux_ = []
          idx_cond = np.where(aux_cond == True)[0]
          if np.where(idx_cond > idx_p)[0].size != 0:
            idx_a = idx_cond[np.where(idx_cond > idx_p)[0][0]]
            aux_.append(idx_a)
          else:
            idx_a = N
            aux_.append(idx_a - 1)
            a = 1
          idx_p = idx_a
          idx_cond = np.where(aux_cond == False)[0]
          if idx_cond.size != 0:
            if np.where(idx_cond > idx_p)[0].size != 0:
              idx_a = idx_cond[np.where(idx_cond > idx_p)[0][0]]
              aux_.append(idx_a - 1)
              idx_p = idx_a -1
            else:
              idx_a = N
              aux_.append(idx_a - 1)
              a = 1
          else:
            idx_a = N
            aux_.append(idx_a - 1)
            a = 1
          Auxidx.append(aux_)
        Auxidx = np.array(Auxidx)
        
        for idx in Auxidx:
            Npeaks = len(np.where((peaks>=idx[0]) & (peaks<=idx[1]))[0])
        return(np.array(Npeaks))