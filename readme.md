# 🧠 Implementación y Evaluación de Filtros Digitales en Python

Este proyecto corresponde a la **Actividad Formativa 3** del curso de *Señales y Sistemas*.  
Su objetivo es diseñar, aplicar y analizar **filtros digitales** pasa-bajos, pasa-altos y pasa-banda utilizando **Python**, **NumPy**, **SciPy** y **Matplotlib**.

---

## 📘 Descripción General

Se genera una señal compuesta por tres frecuencias (30 Hz, 150 Hz y 300 Hz) con ruido blanco.  
Posteriormente se aplican tres tipos de filtros digitales:

| Tipo de Filtro | Método | Parámetros |
|----------------|---------|-------------|
| **Pasa-bajos** | Butterworth | Orden 4, fc = 100 Hz |
| **Pasa-altos** | Chebyshev Tipo I | Orden 4, fc = 120 Hz, rizado 1 dB |
| **Pasa-banda** | FIR (Ventana Hamming) | 129 coeficientes, 100–200 Hz |

El código muestra las respuestas en frecuencia (magnitud y fase), las señales antes y después del filtrado y el cálculo de la **relación señal-ruido (SNR)** para cuantificar la mejora obtenida.

---

## ⚙️ Requisitos

Antes de ejecutar, instala las librerías necesarias:

```bash
pip install numpy scipy matplotlib


Estructura del prooyecto

C:\Users\tabst\OneDrive\programas\Python\filtros
│
├── filtros_digitales.py      # Script principal
├── README.md                 # Descripción del proyecto
└── resultados\               # (Opcional) Gráficas exportadas


Abraham Rubén Tamez Rodríguez
Ingeniería en Desarrollo de Software – 2025