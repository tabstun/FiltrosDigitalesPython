"""
Actividad Formativa 3: Implementación y evaluación de filtros digitales
Alumno: Abraham Rubén Tamez Rodríguez
Materia: SEÑALES Y SISTEMAS (A) (2025-3)
Docente: Ing. Armando Valeriano García García

Descripción:
Este programa genera una señal compuesta con ruido blanco, diseña filtros pasa bajos, pasa altos y pasa bandas (IIR y FIR), 
los aplica a la señal y analiza los resultados en el dominio del tiempo y de la frecuencia. 
También calcula la relación señal/ruido (SNR) para evaluar la efectividad de cada filtro.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


# =====================================================================
# 1. GENERACIÓN DE LA SEÑAL DE PRUEBA
# =====================================================================
# Señal compuesta de tres frecuencias (30, 150 y 300 Hz) con ruido blanco.

fs = 1000.0       # Frecuencia de muestreo [Hz]
T = 2.0           # Duración total [s]
N = int(T * fs)   # Número total de muestras
t = np.arange(N) / fs

# Señal "limpia"
x_clean = (
    0.9 * np.sin(2 * np.pi * 30  * t) +
    0.6 * np.sin(2 * np.pi * 150 * t) +
    0.4 * np.sin(2 * np.pi * 300 * t)
)

# Ruido blanco gaussiano
np.random.seed(42)
noise = 0.8 * np.random.randn(N)

# Señal ruidosa (observada)
x = x_clean + noise


# =====================================================================
# 2. FUNCIONES AUXILIARES (ESPECTRO Y SNR)
# =====================================================================

def spec_db(sig, fs):
    """
    Calcula el espectro de magnitud en dB de una señal sig, usando FFT
    de una sola cara con ventana de Hanning.
    """
    N = len(sig)
    S = np.fft.rfft(sig * np.hanning(N))
    f_ax = np.fft.rfftfreq(N, d=1/fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(S), 1e-12))
    return f_ax, mag_db


def snr_db(x_obs, x_ref):
    """
    Calcula la relación señal/ruido (SNR) en dB tomando como referencia
    x_ref (señal ideal) y x_obs (señal observada o filtrada).
    """
    noise = x_obs - x_ref
    Ps = np.mean(x_ref**2)
    Pn = np.mean(noise**2) if np.mean(noise**2) > 0 else 1e-12
    return 10 * np.log10(Ps / Pn)


# =====================================================================
# 3. ANÁLISIS INICIAL: SEÑAL ORIGINAL EN TIEMPO Y FRECUENCIA
# =====================================================================

# Tiempo
plt.figure()
plt.title("Señal original ruidosa - Tiempo")
plt.plot(t, x)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# Frecuencia
f_orig, X_orig_db = spec_db(x, fs)
plt.figure()
plt.title("Señal original ruidosa - Frecuencia")
plt.plot(f_orig, X_orig_db)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)


# =====================================================================
# 4. DISEÑO DE LOS FILTROS DIGITALES (IIR Y FIR)
# =====================================================================

# 4.1 Filtro pasa-bajos Butterworth (fc = 100 Hz, orden 4)
b_lp, a_lp = signal.butter(N=4, Wn=100.0, btype="low", fs=fs)

# 4.2 Filtro pasa-altos Chebyshev I (fc = 120 Hz, orden 4, rizado 1 dB)
b_hp, a_hp = signal.cheby1(N=4, rp=1, Wn=120.0, btype="high", fs=fs)

# 4.3 Filtro pasa-bandas FIR con ventana Hamming (100–200 Hz, 129 coeficientes)
b_bp = signal.firwin(
    numtaps=129,
    cutoff=[100.0, 200.0],
    pass_zero=False,
    window="hamming",
    fs=fs
)
a_bp = 1.0


# =====================================================================
# 5. RESPUESTA EN FRECUENCIA DE LOS FILTROS
# =====================================================================

# Pasa-bajos Butterworth
w_lp, h_lp = signal.freqz(b_lp, a_lp, worN=2048, fs=fs)
plt.figure()
plt.title("LP Butterworth (fc=100 Hz) - Magnitud")
plt.plot(w_lp, 20 * np.log10(np.maximum(np.abs(h_lp), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.figure()
plt.title("LP Butterworth (fc=100 Hz) - Fase")
plt.plot(w_lp, np.unwrap(np.angle(h_lp)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)

# Pasa-altos Chebyshev I
w_hp, h_hp = signal.freqz(b_hp, a_hp, worN=2048, fs=fs)
plt.figure()
plt.title("HP Chebyshev I (fc=120 Hz, rp=1 dB) - Magnitud")
plt.plot(w_hp, 20 * np.log10(np.maximum(np.abs(h_hp), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.figure()
plt.title("HP Chebyshev I (fc=120 Hz, rp=1 dB) - Fase")
plt.plot(w_hp, np.unwrap(np.angle(h_hp)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)

# Pasa-bandas FIR Hamming
w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=2048, fs=fs)
plt.figure()
plt.title("BP FIR Hamming (100–200 Hz) - Magnitud")
plt.plot(w_bp, 20 * np.log10(np.maximum(np.abs(h_bp), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.figure()
plt.title("BP FIR Hamming (100–200 Hz) - Fase")
plt.plot(w_bp, np.unwrap(np.angle(h_bp)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)


# =====================================================================
# 6. APLICACIÓN DE LOS FILTROS A LA SEÑAL RUIDOSA
# =====================================================================

# Pasa-bajos (fase cero)
y_lp = signal.filtfilt(b_lp, a_lp, x)

# Pasa-altos (fase cero)
y_hp = signal.filtfilt(b_hp, a_hp, x)

# Pasa-bandas FIR (puede usarse filtfilt, aquí se usa filtfilt para fase cero)
y_bp = signal.filtfilt(b_bp, a_bp, x)


# =====================================================================
# 7. COMPARACIÓN EN TIEMPO Y FRECUENCIA + SNR
# =====================================================================

# --- 7.1 Pasa-bajos: comparación y SNR respecto a 30 Hz ---
ref30 = 0.9 * np.sin(2 * np.pi * 30 * t)

plt.figure()
plt.title("Tiempo: original vs filtrada (LP 100 Hz)")
plt.plot(t, x, label="Original", alpha=0.45)
plt.plot(t, y_lp, label="Filtrada (LP 100 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.xlim(0, 0.2)  # zoom 0–0.2 s

f_lp, Y_lp_db = spec_db(y_lp, fs)
plt.figure()
plt.title("Frecuencia: original vs filtrada (LP 100 Hz)")
plt.plot(f_orig, X_orig_db, label="Original", alpha=0.45)
plt.plot(f_lp,   Y_lp_db,   label="Filtrada (LP 100 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()
plt.xlim(0, 500)

print("SNR (dB) original vs 30 Hz:", snr_db(x,    ref30))
print("SNR (dB) LP       vs 30 Hz:", snr_db(y_lp, ref30))


# --- 7.2 Pasa-altos: comparación y SNR respecto a 300 Hz ---
ref300 = 0.4 * np.sin(2 * np.pi * 300 * t)

plt.figure()
plt.title("Tiempo: original vs filtrada (HP 120 Hz)")
plt.plot(t, x,   label="Original", alpha=0.45)
plt.plot(t, y_hp, label="Filtrada (HP 120 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.xlim(0, 0.2)

f_hp_sig, Y_hp_db = spec_db(y_hp, fs)
plt.figure()
plt.title("Frecuencia: original vs filtrada (HP 120 Hz)")
plt.plot(f_orig, X_orig_db, label="Original", alpha=0.45)
plt.plot(f_hp_sig, Y_hp_db, label="Filtrada (HP 120 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()
plt.xlim(0, 500)

print("SNR (dB) original vs 300 Hz:", snr_db(x,    ref300))
print("SNR (dB) HP       vs 300 Hz:", snr_db(y_hp, ref300))


# --- 7.3 Pasa-bandas: comparación y SNR respecto a 150 Hz ---
ref150 = 0.6 * np.sin(2 * np.pi * 150 * t)

plt.figure()
plt.title("Tiempo: original vs filtrada (BP 100–200 Hz)")
plt.plot(t, x,   label="Original", alpha=0.45)
plt.plot(t, y_bp, label="Filtrada (BP 100–200 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.xlim(0, 0.2)

f_bp_sig, Y_bp_db = spec_db(y_bp, fs)
plt.figure()
plt.title("Frecuencia: original vs filtrada (BP 100–200 Hz)")
plt.plot(f_orig, X_orig_db, label="Original", alpha=0.45)
plt.plot(f_bp_sig, Y_bp_db, label="Filtrada (BP 100–200 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()
plt.xlim(0, 500)

print("SNR (dB) original vs 150 Hz:", snr_db(x,    ref150))
print("SNR (dB) BP       vs 150 Hz:", snr_db(y_bp, ref150))


# =====================================================================
# 8. MOSTRAR TODAS LAS GRÁFICAS
# =====================================================================
plt.show()
