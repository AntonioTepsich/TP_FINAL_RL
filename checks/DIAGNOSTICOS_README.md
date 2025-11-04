# 🔬 Plan de Diagnóstico y Mejora - Flappy Bird PPO

## 📋 Checklist Priorizada

### ✅ Scripts Creados

| Script | Tiempo | Propósito |
|--------|--------|-----------|
| `check_obs.py` | 30s | Verificar si son vectores o píxeles |
| `check_gae.py` | 5s | Test matemático de GAE |
| `test_rapido.py` | 1-2min | Diagnóstico rápido con métricas PPO |
| `train_improved.py` | 15-20min | Entrenamiento completo con todas las fixes |

---

## 🚀 Plan de Acción (Orden Sugerido)

### **PASO 1: Verificar Observaciones** (30 segundos)
```bash
python check_obs.py
```

**Qué buscar:**
- ✅ Si son **vectores** (12 features): Perfecto, continúa al Paso 2
- ❌ Si son **píxeles**: Necesitas usar CNN (en `ppo.py`) o wrapper para extraer features

**Acción si son píxeles:**
- Opción A: Usa el env con render_mode diferente para obtener features
- Opción B: Crea wrapper que extrae (posición bird, velocidad, distancia a pipes)

---

### **PASO 2: Verificar GAE** (5 segundos)
```bash
python check_gae.py
```

**Qué buscar:**
- ✅ Todos los checks pasan
- ❌ Si falla algún check: revisar implementación de GAE

---

### **PASO 3: Test Rápido con Diagnósticos** (1-2 minutos)
```bash
python test_rapido.py
```

**Qué observar en los logs:**

#### 📊 **POLICY Metrics:**
| Métrica | Rango Bueno | Acción si Fuera de Rango |
|---------|-------------|--------------------------|
| **Ratio** | 0.8 - 1.2 | <0.8 o >1.3: Bajá LR |
| **KL Div** | < 0.02 | >0.03: Bajá LR o epochs |
| **Clip Frac** | 0.1 - 0.4 | <0.05: Subí LR / >0.5: Bajá LR |

#### 📈 **VALUE Metrics:**
| Métrica | Rango Bueno | Acción si Fuera de Rango |
|---------|-------------|--------------------------|
| **Explained Var** | > 0.3 | <0.2: Subí `vf_coef` o capacidad red |

#### 🎲 **EXPLORATION Metrics:**
| Métrica | Rango Bueno | Acción si Fuera de Rango |
|---------|-------------|--------------------------|
| **Entropy** | 0.05 - 0.3 | <0.01: Subí `ent_coef` |

---

### **PASO 4: Ajustar Hiperparámetros**

Basado en los diagnósticos del Paso 3, editá `train_improved.py`:

```python
# CONFIGURACIÓN (líneas 132-140)
N_ENVS = 16          # 8-24 según tu CPU/GPU
T = 256              # 128-512 (más = mejor GAE, menos = updates más frecuentes)

LR_START = 3e-4      # Bajá a 1e-4 si ratio/KL muy alto
LR_END = 1e-5
ENT_START = 0.02     # Subí a 0.03-0.05 si entropía cae muy rápido
ENT_END = 0.005

# En agent (línea 154)
clip_eps=0.2,        # Standard
ent_coef=ENT_START,  
vf_coef=0.5,         # Subí a 1.0 si explained_var < 0.2
```

---

### **PASO 5: Entrenamiento Completo** (15-20 minutos)
```bash
python train_improved.py
```

**Monitorear durante el entrenamiento:**

1. **Primeros 100k steps:**
   - Reward debería pasar de ~0 a 10-20
   - Entropy bajando de 0.02 → 0.015
   - Explained Var subiendo a >0.5

2. **500k steps:**
   - Reward ~50-100
   - Entropy ~0.01
   - KL estable <0.02

3. **1M steps:**
   - Reward >100-200
   - Política estable

---

## 🔧 Troubleshooting Común

### Problema 1: No aprende (reward estancado)
**Síntomas:** Reward se queda en 0-5 por mucho tiempo

**Checks:**
```python
# En train_improved.py, agregá después del rollout:
print(f"Sample rewards: {rews_t[:,0][:10]}")  # Ver si hay rewards positivos
print(f"Sample actions: {acts_t[:50]}")        # Ver distribución de acciones
```

**Soluciones:**
- Si >90% acciones son la misma: Subí `ent_coef` a 0.03-0.05
- Si rewards todos negativos: Verificá que el env da +1 por sobrevivir
- Si explained_var <0: Red no aprende, aumentá capacidad (hidden=256→512)

---

### Problema 2: Inestable (reward sube y baja mucho)
**Síntomas:** Reward llega a 50, luego cae a 10, sube a 80, etc.

**Checks:**
- Mirá KL div y ratio en logs
- Si KL >0.03 o ratio >1.5: Updates muy agresivos

**Soluciones:**
1. Bajá LR: `3e-4 → 1e-4`
2. Reducí epochs: `4 → 3`
3. Aumentá `N_ENVS` para más estabilidad

---

### Problema 3: Aprende pero no llega lejos
**Síntomas:** Reward estable en 30-50 pero no sube más

**Checks:**
- Entropy muy baja (<0.005): Política muy determinista, no explora
- Clip frac muy alto (>0.6): Updates muy conservadores

**Soluciones:**
1. Si entropy baja: Ajustá schedule para que baje más lento
2. Aumentá `T` (256→512) para mejor estimación GAE
3. Probá diferentes seeds

---

## 📊 Mejoras Aplicadas vs Original

| Aspecto | Original (`train_vector.py`) | Mejorado (`train_improved.py`) |
|---------|------------------------------|--------------------------------|
| **Normalización** | ❌ No | ✅ Online normalization |
| **Métricas PPO** | ❌ Solo loss | ✅ KL, ratio, clip_frac, EV |
| **Schedules** | ❌ Fijo | ✅ LR cosine, entropy linear |
| **Diagnósticos** | ❌ Básico | ✅ Warnings automáticos |
| **Value clipping** | ❌ No | ✅ Sí |
| **Logging** | ⚠️ Básico | ✅ Completo con interpretación |

---

## 🎯 Resultados Esperados

### Con observaciones vectoriales (12 features):
- **100k steps** (~2 min): Reward ~10-30
- **500k steps** (~8 min): Reward ~50-100
- **1M steps** (~15 min): Reward >100-200

### Señales de éxito:
- ✅ Explained Variance >0.5 en primeros 200k steps
- ✅ KL div estable <0.02
- ✅ Entropy baja gradualmente (no colapsa en 0)
- ✅ Reward crece monotónicamente (con ruido)

---

## 🔬 Experimentos Adicionales (Opcional)

### A) Probar diferentes T (rollout length)
```bash
# Editá en train_improved.py línea 134
T = 128   # Más updates, señales frescas
T = 512   # Mejor GAE, más estable
```

### B) Probar diferentes N_ENVS
```bash
N_ENVS = 8    # Más rápido, menos estable
N_ENVS = 24   # Más lento, más estable
```

### C) Comparar con/sin normalización
```bash
# En train_improved.py línea 149
vec = VecEnv(n_envs=N_ENVS, normalize=False)  # Sin normalización
```

---

## 📝 Notas Importantes

1. **Reproducibilidad**: Seeds fijas en `reset(seed=i)` (ya implementado)

2. **Guardado de modelos**:
   - `best_model_improved.pt`: Mejor modelo por reward
   - `checkpoint_improved_250000.pt`: Checkpoints cada 250k

3. **Cargar modelo guardado**:
```python
model = VectorActorCritic(obs_dim=12, n_actions=2)
model.load_state_dict(torch.load('best_model_improved.pt'))
model.eval()
```

4. **Si vas a píxeles**:
   - Necesitás CNN (ya en `ppo.py`)
   - Cambiá obs/255.0
   - Esperá 2-3 horas en vez de 15 min
   - Ajustá `ConvEncoder.out_dim` según resolución

---

## ✅ Checklist Final Antes de 1M Steps

- [ ] `check_obs.py` confirma vectores 12D
- [ ] `check_gae.py` pasa todos los tests
- [ ] `test_rapido.py` muestra métricas razonables
- [ ] Explained Var >0.3 en test rápido
- [ ] KL <0.03 en test rápido
- [ ] Ajustaste hiperparámetros según diagnósticos
- [ ] GPU/CPU libre para 15-20 min

**Si todos ✅, corre:**
```bash
python train_improved.py
```

---

## 🆘 Si Nada Funciona

1. Compartí output de `check_obs.py`
2. Compartí primeros logs de `test_rapido.py` (métricas PPO)
3. Compartí gráfica de reward vs steps (aunque sea solo valores)

Podemos debugear desde ahí con info concreta.
