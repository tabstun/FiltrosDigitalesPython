import numpy as np
import matplotlib.pyplot as plt

# Frecuencia de muestreo y duración
fs = 1000.0      # Hz
T = 2.0           # segundos
N = int(T * fs)   # número total de muestras
t = np.arange(N) / fs

# Señal "limpia": tres frecuencias diferentes (30, 150 y 300 Hz)
x_clean = (0.9 * np.sin(2*np.pi*30*t) +
           0.6 * np.sin(2*np.pi*150*t) +
           0.4 * np.sin(2*np.pi*300*t))

# Añadimos ruido blanco (semilla fija para reproducibilidad)
np.random.seed(42)
noise = 0.8 * np.random.randn(N)
x = x_clean + noise # señal total: señal limpia + ruido

# ----- Gráficas -----
# Dominio del tiempo
plt.figure()
plt.title("Señal original (tiempo)")
plt.plot(t, x)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# Dominio de la frecuencia -- Transformada Rápida de Fourier (FFT) ---
X = np.fft.rfft(x * np.hanning(N)) # ventana Hanning para suavizar el espectro
f = np.fft.rfftfreq(N, d=1/fs)

plt.figure()
plt.title("Señal original (frecuencia)")
plt.plot(f, 20*np.log10(np.maximum(np.abs(X), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.show()


# ===== Paso 3A: Diseño del filtro pasa-bajos =====
from scipy import signal

# Diseño del filtro Butterworth de orden 4
b_lp, a_lp = signal.butter(N=4, Wn=100.0, btype='low', fs=fs)

# Respuesta en frecuencia del filtro 
w, h = signal.freqz(b_lp, a_lp, worN=2048, fs=fs)

plt.figure()
plt.title("LP Butterworth (fc=100 Hz) - Magnitud")
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

plt.figure()
plt.title("LP Butterworth (fc=100 Hz) - Fase")
plt.plot(w, np.unwrap(np.angle(h)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.grid(True)

# plt.show() los converti el comando en comentario para que solo muestre al final 


# ===== Paso 3B: Aplicar el filtro pasa-bajos a la señal =====
from scipy import signal

# Aplicar el filtro con filtfilt (fase cero, evita desfase)
y_lp = signal.filtfilt(b_lp, a_lp, x)

# --- Señal filtrada en el tiempo ---
plt.figure()
plt.title("Señal filtrada (pasa-bajos) - Tiempo")
plt.plot(t, y_lp)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

# --- Espectro de la señal filtrada ---
Y_lp = np.fft.rfft(y_lp * np.hanning(N))
f = np.fft.rfftfreq(N, d=1/fs)

plt.figure()
plt.title("Señal filtrada (pasa-bajos) - Frecuencia")
plt.plot(f, 20*np.log10(np.maximum(np.abs(Y_lp), 1e-12)))
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

# plt.show() los converti el comando en comentario para que solo muestre al final 


# ===== Comparación directa y SNR del LP (100 Hz) =====
import numpy as np
import matplotlib.pyplot as plt

# 1) Tiempo: original vs filtrada con zoom
plt.figure()
plt.title("Tiempo: original vs filtrada (LP 100 Hz)")
plt.plot(t, x, label="Original", alpha=0.45)
plt.plot(t, y_lp, label="Filtrada (LP 100 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.grid(True); plt.legend()
plt.xlim(0, 0.2)  # zoom 0–0.2 s para ver la suavización

# 2) Frecuencia: espectros superpuestos en dB, la cual es la función auxiliar para obtener espectro en dB
def spec_db(sig, fs):
    N = len(sig)
    S = np.fft.rfft(sig * np.hanning(N))
    f = np.fft.rfftfreq(N, d=1/fs)
    return f, 20*np.log10(np.maximum(np.abs(S), 1e-12))

# Espectros superpuestos
f_ax, X_db = spec_db(x, fs)
_,    Y_db = spec_db(y_lp, fs)

plt.figure()
plt.title("Frecuencia: original vs filtrada (LP 100 Hz)")
plt.plot(f_ax, X_db, label="Original", alpha=0.45)
plt.plot(f_ax, Y_db, label="Filtrada (LP 100 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud [dB]"); plt.grid(True); plt.legend()
plt.xlim(0, 500)

# 3) SNR respecto a 30 Hz (la componente que debe preservarse)
def snr_db(x_obs, x_ref):
    noise = x_obs - x_ref
    Ps = np.mean(x_ref**2)
    Pn = np.mean(noise**2) if np.mean(noise**2) > 0 else 1e-12
    return 10*np.log10(Ps/Pn)

ref30 = 0.9*np.sin(2*np.pi*30*t)
print("SNR (dB) original vs 30 Hz:", snr_db(x, ref30))
print("SNR (dB) LP      vs 30 Hz:", snr_db(y_lp, ref30))

# plt.show() los converti el comando en comentario para que solo muestre al final 

# ===== Pasa-altos Chebyshev I (fc=120 Hz, orden 4, rizado 1 dB) =====
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Diseño Chebyshev tipo I (orden 4, rizado de 1 dB)
b_hp, a_hp = signal.cheby1(N=4, rp=1, Wn=120.0, btype='high', fs=fs)

# Respuesta en frecuencia
w_hp, h_hp = signal.freqz(b_hp, a_hp, worN=2048, fs=fs)
plt.figure(); plt.title("HP Chebyshev I (fc=120 Hz, rp=1 dB) - Magnitud")
plt.plot(w_hp, 20*np.log10(np.maximum(np.abs(h_hp), 1e-12))); plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]"); plt.grid(True)

plt.figure(); plt.title("HP Chebyshev I (fc=120 Hz, rp=1 dB) - Fase")
plt.plot(w_hp, np.unwrap(np.angle(h_hp))); plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]"); plt.grid(True)


# ===== Filtrado y comparación (tiempo y frecuencia) =====
y_hp = signal.filtfilt(b_hp, a_hp, x)   # fase cero

# Tiempo con superposición (zoom para notar diferencias)
plt.figure(); plt.title("Tiempo: original vs filtrada (HP 120 Hz)")
plt.plot(t, x,  label="Original", alpha=0.45)
plt.plot(t, y_hp, label="Filtrada (HP 120 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.grid(True); plt.legend()
plt.xlim(0, 0.2)

# Frecuencia con superposición en dB
def spec_db(sig, fs):
    N = len(sig); S = np.fft.rfft(sig * np.hanning(N))
    f_ax = np.fft.rfftfreq(N, d=1/fs)
    return f_ax, 20*np.log10(np.maximum(np.abs(S), 1e-12))

# Comparación espectral
f_ax, X_db  = spec_db(x, fs)
_,    Yh_db = spec_db(y_hp, fs)

plt.figure(); plt.title("Frecuencia: original vs filtrada (HP 120 Hz)")
plt.plot(f_ax, X_db,  label="Original", alpha=0.45)
plt.plot(f_ax, Yh_db, label="Filtrada (HP 120 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud [dB]")
plt.grid(True); plt.legend(); plt.xlim(0, 500)


# ===== SNR: ¿mejora al aislar altas frecuencias (300 Hz)? =====
def snr_db(x_obs, x_ref):
    noise = x_obs - x_ref
    Ps = np.mean(x_ref**2); Pn = np.mean(noise**2) if np.mean(noise**2)>0 else 1e-12
    return 10*np.log10(Ps/Pn)

ref300 = 0.4*np.sin(2*np.pi*300*t)

print("SNR (dB) original vs 300 Hz:", snr_db(x,    ref300))
print("SNR (dB) HP       vs 300 Hz:", snr_db(y_hp, ref300))


# plt.show() los converti el comando en comentario para que solo muestre al final 

# ===== Pasa-bandas FIR (100–200 Hz, 129 coeficientes, ventana Hamming) =====
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Diseño del filtro FIR con 129 coeficientes
b_bp = signal.firwin(numtaps=129, cutoff=[100.0, 200.0], pass_zero=False,
                     window='hamming', fs=fs)
a_bp = 1.0

# Respuesta en frecuencia del FIR
w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=2048, fs=fs)
plt.figure(); plt.title("BP FIR Hamming (100–200 Hz) - Magnitud")
plt.plot(w_bp, 20*np.log10(np.maximum(np.abs(h_bp), 1e-12)))
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud [dB]"); plt.grid(True)

plt.figure(); plt.title("BP FIR Hamming (100–200 Hz) - Fase")
plt.plot(w_bp, np.unwrap(np.angle(h_bp)))
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Fase [rad]"); plt.grid(True)

# ===== Aplicar el filtro y comparar =====
y_bp = signal.filtfilt(b_bp, a_bp, x)

# Tiempo con zoom
plt.figure(); plt.title("Tiempo: original vs filtrada (BP 100–200 Hz)")
plt.plot(t, x,  label="Original", alpha=0.45)
plt.plot(t, y_bp, label="Filtrada (BP 100–200 Hz)", linewidth=1.5)
plt.xlabel("Tiempo [s]"); plt.ylabel("Amplitud"); plt.grid(True); plt.legend()
plt.xlim(0, 0.2)

# Frecuencia superpuesta
def spec_db(sig, fs):
    N = len(sig); S = np.fft.rfft(sig * np.hanning(N))
    f_ax = np.fft.rfftfreq(N, d=1/fs)
    return f_ax, 20*np.log10(np.maximum(np.abs(S), 1e-12))

# Comparación espectral
f_ax, X_db = spec_db(x, fs)
_,    Yb_db = spec_db(y_bp, fs)

plt.figure(); plt.title("Frecuencia: original vs filtrada (BP 100–200 Hz)")
plt.plot(f_ax, X_db, label="Original", alpha=0.45)
plt.plot(f_ax, Yb_db, label="Filtrada (BP 100–200 Hz)", linewidth=1.5)
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud [dB]"); plt.grid(True); plt.legend()
plt.xlim(0, 500)

# ===== SNR respecto a 150 Hz =====
def snr_db(x_obs, x_ref):
    noise = x_obs - x_ref
    Ps = np.mean(x_ref**2)
    Pn = np.mean(noise**2) if np.mean(noise**2)>0 else 1e-12
    return 10*np.log10(Ps/Pn)


# SNR con respecto a la componente de 150 Hz
ref150 = 0.6*np.sin(2*np.pi*150*t)
print("SNR (dB) original vs 150 Hz:", snr_db(x, ref150))
print("SNR (dB) BP        vs 150 Hz:", snr_db(y_bp, ref150))

plt.show()
