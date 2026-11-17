# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:45:51 2026

@author: USUARIO
"""


import numpy as np
from scipy import signal
import scipy.io as sio
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter
from scipy.interpolate import CubicSpline




fs_ecgconruido = 1000  # Hz


print(sio.whosmat('ECG_TP4.mat'))

mat_struct = sio.loadmat("ECG_TP4.mat")
ecg_conruido = mat_struct["ecg_lead"].flatten().astype(float)
t_ecgconruido = np.arange(len(ecg_conruido)) / fs_ecgconruido

N = len(ecg_conruido)

ecg_sr= np.load('ecg_sin_ruido.npy')

qrs_det=mat_struct['qrs_detections'].flatten()




#%%Línea de base por mediana 
b200 = signal.medfilt(ecg_conruido, kernel_size=201)
b600 = signal.medfilt(b200, kernel_size=601)

ecg_filtrado_med = ecg_conruido - b600

#%% por splines cúbicos

start_win = qrs_det - 60
end_win = qrs_det - 40

pq_hat = np.array([np.mean(ecg_conruido[a:b]) for a, b in zip(start_win, end_win)])
pq_that = (start_win + end_win) // 2

cs = CubicSpline(pq_that, pq_hat)
n = np.arange(len(ecg_conruido))
b_spline = cs(n)

ecg_filtrado_spl = ecg_conruido - b_spline

plt.figure(figsize=(14, 4))
plt.plot(t_ecgconruido, ecg_conruido, label='ECG')
plt.plot(t_ecgconruido, ecg_filtrado_spl, label='ECG filtrado')
plt.plot(t_ecgconruido, b_spline, label='linea de base con spline')
plt.plot(pq_that / fs_ecgconruido, pq_hat, 'g.', markersize=3, label='PQ')
plt.plot(qrs_det / fs_ecgconruido, ecg_conruido[qrs_det], 'rx', markersize=3, label='QRS')
plt.xlabel('Tiempo [s]'); plt.ylabel('Amplitud')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

#%% Comparación


zonas = [(4000, 8000), (100000, 120000), (720000, 740000)]

for ini, fin in zonas:
    tz = t_ecgconruido[ini:fin]
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)


    ax[0].plot(tz, ecg_conruido[ini:fin], color='gray', alpha=0.6, label='ECG sin filtrar')
    ax[0].plot(tz, b600[ini:fin], color='tab:orange', linewidth=2, label=r'$\hat{b}$ mediana (201 → 601)')
    ax[0].plot(tz, b_spline[ini:fin], color='tab:green', linewidth=2, label=r'$\hat{b}$ spline cúbico')
    en_zona = (pq_that >= ini) & (pq_that < fin)
    ax[0].plot(pq_that[en_zona] / fs_ecgconruido, pq_hat[en_zona], 'ko', markersize=4, label='puntos PQ')
    ax[0].set_ylabel('Amplitud'); ax[0].set_title('Estimación de la línea de base')
    ax[0].grid(alpha=0.3); ax[0].legend(loc='upper right')


    ax[1].plot(tz, ecg_filtrado_med[ini:fin], color='tab:orange', alpha=0.8, label=r'$\hat{x} = s - \hat{b}_{med}$')
    ax[1].plot(tz, ecg_filtrado_spl[ini:fin], color='tab:green', alpha=0.8, label=r'$\hat{x} = s - \hat{b}_{spline}$')
    ax[1].axhline(0, color='k', linewidth=0.8)
    ax[1].set_xlabel('Tiempo [s]'); ax[1].set_ylabel('Amplitud')
    ax[1].set_title('ECG con línea de base corregida')
    ax[1].grid(alpha=0.3); ax[1].legend(loc='upper right')
    plt.tight_layout()
plt.show()


dif_b = b600 - b_spline
print('\n--- Comparación de líneas de base ---')
print(f'RMS(b_mediana - b_spline) = {np.sqrt(np.mean(dif_b**2)):.2f}')
print(f'Máx |b_mediana - b_spline| = {np.max(np.abs(dif_b)):.2f}')


f_b, P_med = signal.welch(b600, fs=fs_ecgconruido, nperseg=2**16)
_, P_spl = signal.welch(b_spline, fs=fs_ecgconruido, nperseg=2**16)

plt.figure(figsize=(10, 4))
plt.semilogy(f_b, P_med, color='tab:orange', label=r'$\hat{b}$ mediana')
plt.semilogy(f_b, P_spl, color='tab:green', label=r'$\hat{b}$ spline')
plt.axvline(0.5, color='k', linestyle='--', label='0.5 Hz')
plt.xlim(0, 5)
en_rango = f_b <= 5
plt.ylim(min(P_med[en_rango].min(), P_spl[en_rango].min()) / 10, None)
plt.xlabel('Frecuencia [Hz]'); plt.ylabel('PSD')
plt.title('Densidad espectral de las líneas de base estimadas')
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.show()

#%%Filtro adaptado

qrs_p1 = mat_struct['qrs_pattern1'].flatten().astype(float)
qrs_p1 = qrs_p1 - np.mean(qrs_p1)             # patrón con media cero


qrs_p1_matched = signal.lfilter(qrs_p1[::-1], 1, ecg_conruido)


qrs_p1_matched = np.abs(qrs_p1_matched - np.mean(qrs_p1_matched)) / np.std(qrs_p1_matched)
ecg_conruido_norm = (ecg_conruido - np.mean(ecg_conruido)) / np.std(ecg_conruido)

# Gráfico
plt.figure(figsize=(14, 4))
plt.plot(ecg_conruido_norm, label='ECG', alpha=0.7)
plt.plot(qrs_p1_matched[70:], label='ECG filtrado adaptado')
plt.plot(qrs_det, ecg_conruido_norm[qrs_det], 'rx', label='qrs_det')
plt.xlabel('Muestras'); plt.ylabel('Amplitud normalizada')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

#%%
#filtro pasabajos

sos = signal.iirdesign(wp=2, ws=20, gpass=1, gstop=40,
                       analog=False, ftype='butter', output='sos', fs=1000)

ecgfilt_filt= signal.sosfiltfilt(sos, qrs_p1_matched)
picos, props = signal.find_peaks(qrs_p1_matched, height=1, distance=500)
picos_filt, props_filt = signal.find_peaks(ecgfilt_filt, height=0.5, distance=500)

plt.figure(figsize=(14, 4))
plt.plot(qrs_p1_matched, label='ECG filtrado adaptado')
plt.plot(ecgfilt_filt, label='pasabajos al filtrado')
plt.plot(picos, qrs_p1_matched[picos], 'rx', label='picos')
plt.plot(picos_filt, ecgfilt_filt[picos_filt], 'ko', mfc='none', label='picos pasabajos')
plt.xlabel('Muestras'); plt.ylabel('Amplitud normalizada')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()


#%% detector y qrs_detections


# Demora del filtro adaptado
demora = len(qrs_p1) - 1 - np.argmax(np.abs(qrs_p1))   

tol = 100   

def evaluar(det, ref, tol):
    dif = np.abs(det[:, None] - ref[None, :])    
    TP = np.sum(np.min(dif, axis=0) <= tol)       
    FN = len(ref) - TP                            
    FP = np.sum(np.min(dif, axis=1) > tol)       
    S = TP / (TP + FN)                            
    PPV = TP / (TP + FP)                      
    print(f'TP={TP}  FP={FP}  FN={FN}  S={100*S:.2f}%  PPV={100*PPV:.2f}%')

print('Filtro adaptado:')
evaluar(picos - demora, qrs_det, tol)
print('Filtro adaptado + pasabajos:')
evaluar(picos_filt - demora, qrs_det, tol)


picos_300, _ = signal.find_peaks(qrs_p1_matched, height=1, distance=300)
print('Filtro adaptado (distance=300):')
evaluar(picos_300 - demora, qrs_det, tol)

plt.figure(figsize=(14, 4))
plt.plot(ecg_filtrado_med, color='gray', label='ECG')
plt.plot(qrs_det, ecg_filtrado_med[qrs_det], 'rx', label='qrs_detections')
plt.plot(picos_300 - demora, ecg_filtrado_med[picos_300 - demora], 'ko', mfc='none', label='detecciones')
plt.xlabel('Muestras'); plt.ylabel('Amplitud')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()