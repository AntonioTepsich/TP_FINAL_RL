# TP Final Reinforcement Learning - PPO-Clip en Flappy Bird

**Autores:** Michelle Chloe Berezovsky y Antonio Santiago Tepsich
**Fecha:** 2025  
**Institución:** Universidad de San Andrés

---

## 📋 Descripción

Este proyecto implementa un agente de Reinforcement Learning utilizando el algoritmo **PPO-Clip (Proximal Policy Optimization)** para resolver el juego **Flappy Bird**. El agente aprende a jugar de manera autónoma mediante interacción con el entorno y optimización de políticas.

### Características Principales
- **Algoritmo:** PPO-Clip con entropía adaptativa
- **Entorno:** Flappy Bird (Gymnasium)
- **Red Neuronal:** Actor-Critic con capas compartidas
- **Normalización:** Observaciones normalizadas online
- **Logging:** TensorBoard para monitoreo de entrenamiento
- **Vectorización:** Entrenamiento paralelo con múltiples entornos

---

## 🗂️ Estructura del Proyecto

```
TP_FINAL_RL/
├── ppo.py                      # Implementación del algoritmo PPO-Clip
├── train_vector_improved.py    # Script principal de entrenamiento
├── watch_play.py              # Visualización del agente entrenado
├── tensorboard_logger.py      # Sistema de logging para TensorBoard
├── requirements.txt           # Dependencias del proyecto
├── best_model_improved.pt     # Modelo entrenado (checkpoint)
├── checks/                    # Diagnósticos y verificaciones
│   ├── check_obs.py
│   └── DIAGNOSTICOS_README.md
└── runs/                      # Logs de TensorBoard
```

---

## 🚀 Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone https://github.com/AntonioTepsich/TP_FINAL_RL.git
cd TP_FINAL_RL
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- Flappy Bird Gymnasium >= 0.3.0
- TensorBoard >= 2.14.0
- NumPy >= 1.24.0

---

## 🎮 Uso

### Entrenar un Nuevo Modelo

Para entrenar un agente desde cero:

```bash
python train_vector_improved.py
```

**Configuración de entrenamiento:**
- **Entornos paralelos:** 8
- **Pasos por iteración:** 2048
- **Épocas por actualización:** 4
- **Learning rate inicial:** 3e-4
- **Clip epsilon:** 0.2
- **Entropía adaptativa:** Sí

El entrenamiento guardará:
- Checkpoints en `best_model_improved.pt`
- Logs de TensorBoard en `runs/`

### Visualizar el Entrenamiento

Para monitorear el progreso en tiempo real:

```bash
tensorboard --logdir=runs
```

Luego abre tu navegador en `http://localhost:6006`

### Evaluar el Modelo Entrenado

Para ver al agente jugar:

```bash
python watch_play.py
```

**Parámetros configurables** (en el archivo):
- `episodes`: Número de episodios a visualizar (default: 5)
- `fps`: Velocidad de renderizado (default: 60)
- `debug`: Mostrar información de debugging (default: True)

---

## 📊 Componentes Principales

### PPO-Clip (`ppo.py`)
Implementación del algoritmo Proximal Policy Optimization con:
- Clipping de probabilidades para estabilidad
- Generalised Advantage Estimation (GAE)
- Optimización de Actor y Critic simultáneos

### Red Neuronal Actor-Critic
Arquitectura de la red:
```
Input (180 observaciones) 
    ↓
Capa Compartida (256 unidades) → ReLU
    ↓
Capa Compartida (256 unidades) → ReLU
    ├─→ Policy Head (2 acciones)
    └─→ Value Head (1 valor)
```

### Normalización de Observaciones
Wrapper personalizado que mantiene estadísticas móviles (media y varianza) para normalizar observaciones durante el entrenamiento, mejorando la estabilidad del aprendizaje.

---

## 📈 Resultados

El modelo aprende progresivamente a:
1. **Evitar colisiones** con las tuberías
2. **Mantener altura óptima** en el juego
3. **Maximizar la recompensa** acumulada

Los mejores modelos logran superar múltiples obstáculos consecutivamente.

---

## 🛠️ Diagnósticos

La carpeta `checks/` contiene herramientas de diagnóstico:
- `check_obs.py`: Verificación de dimensiones de observaciones
- `DIAGNOSTICOS_README.md`: Guía de troubleshooting