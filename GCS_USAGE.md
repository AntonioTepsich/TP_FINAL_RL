# Guía de Uso: Integración con Google Cloud Storage

Esta guía explica cómo usar el sistema de entrenamiento de PPO Flappy Bird con integración a Google Cloud Storage (GCS), permitiendo entrenar en Google Colab y guardar todos los resultados de forma automática y organizada en tu bucket.

## 📋 Tabla de Contenidos

1. [Instalación](#instalación)
2. [Configuración Básica](#configuración-básica)
3. [Uso Individual](#uso-individual)
4. [Búsqueda de Hiperparámetros](#búsqueda-de-hiperparámetros)
5. [Visualizar Resultados](#visualizar-resultados)
6. [Estructura del Bucket](#estructura-del-bucket)
7. [Recuperación de Experimentos](#recuperación-de-experimentos)

---

## 📦 Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Autenticación con GCP (en Colab)

```python
from google.colab import auth
auth.authenticate_user()
```

O si prefieres usar una service account key:

```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/your/key.json'
```

### 3. Verificar Conexión

```python
from google.cloud import storage

PROJECT_ID = "quiet-sum-477223-g3"
BUCKET = "ppo-flappy-bird"

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET)

# Listar contenido
for blob in client.list_blobs(BUCKET, max_results=5):
    print(blob.name)
```

---

## ⚙️ Configuración Básica

### Opción 1: Usar Archivo de Configuración (Recomendado)

Edita `config_template.yaml`:

```yaml
# Configuración básica
n_envs: 16
total_steps: 1000000
hidden_size: 256

# GCS
gcs_bucket: ppo-flappy-bird
gcs_project: quiet-sum-477223-g3

# Hiperparámetros
lr_start: 0.0003
lr_end: 0.00001
ent_start: 0.02
ent_end: 0.005
```

### Opción 2: Línea de Comandos

Pasa los argumentos directamente al script:

```bash
python train_vector_improved.py \
    --gcs-bucket ppo-flappy-bird \
    --gcs-project quiet-sum-477223-g3 \
    --lr-start 0.0003 \
    --hidden-size 256
```

---

## 🚀 Uso Individual

### Correr un Experimento Simple

**Con archivo de configuración:**

```bash
python run_experiment.py --config config_template.yaml
```

**Directamente con train_vector_improved.py:**

```bash
python train_vector_improved.py \
    --gcs-bucket ppo-flappy-bird \
    --gcs-project quiet-sum-477223-g3 \
    --total-steps 1000000
```

### Monitorear el Progreso

Durante el entrenamiento verás:
- ✅ Checkpoints guardados localmente cada 250k steps
- ☁️ Uploads automáticos a GCS en background
- 📊 Logs de TensorBoard generándose

### Lo que se Guarda Automáticamente

1. **Checkpoints** (`/checkpoints/`):
   - `best_model_improved.pt` - Mejor modelo (solo weights)
   - `best_model_improved_full.pt` - Modelo + estadísticas de normalización
   - `checkpoint_250k.pt`, `checkpoint_500k.pt`, etc.

2. **Configuración** (`config.json`):
   - Todos los hiperparámetros usados
   - Timestamp del experimento
   - ID único del experimento

3. **Métricas** (`/metrics/final_metrics.json`):
   - Reward final y máximo
   - Score del juego (tubos pasados)
   - Tiempo de entrenamiento
   - Steps por segundo

4. **Logs de TensorBoard** (`/tensorboard/`):
   - Events files completos
   - Histogramas de gradientes
   - Gráficos de métricas

---

## 🔍 Búsqueda de Hiperparámetros

### 1. Configurar el Search Space

Edita `search_config_example.yaml`:

```yaml
strategy: random  # o 'grid'
n_trials: 20      # número de experimentos (para random search)
seed: 42          # reproducibilidad

search_space:
  lr_start:
    - 0.0001
    - 0.0003
    - 0.0005

  hidden_size:
    - 128
    - 256
    - 512

  clip_epsilon:
    - 0.1
    - 0.2
    - 0.3
```

### 2. Ejecutar la Búsqueda

```bash
python run_experiment.py \
    --config config_template.yaml \
    --search \
    --search-config search_config_example.yaml
```

### 3. Estrategias de Búsqueda

**Random Search (Recomendado):**
- Prueba combinaciones aleatorias
- Más eficiente para espacios grandes
- Configura `n_trials` (ej: 10-50)

**Grid Search:**
- Prueba TODAS las combinaciones
- Solo para espacios pequeños
- Cuidado: puede tomar mucho tiempo

### Ejemplo Práctico en Colab

```python
# En una celda de Colab
!python run_experiment.py \
    --config config_template.yaml \
    --search \
    --search-config search_config_example.yaml \
    --output-dir search_results

# Ver progreso en otra celda (ejecutar periódicamente)
!python view_results.py --bucket ppo-flappy-bird --compare --top 5
```

---

## 📊 Visualizar Resultados

### Listar Todos los Experimentos

```bash
python view_results.py --bucket ppo-flappy-bird --list
```

### Comparar Experimentos

```bash
# Ver top 10 por reward
python view_results.py --bucket ppo-flappy-bird --compare --top 10

# Ver todos, ordenados por score
python view_results.py --bucket ppo-flappy-bird --compare --sort-by best_score --top 0

# Exportar a CSV
python view_results.py --bucket ppo-flappy-bird --compare --export-csv --output results.csv
```

### Ver Detalles de un Experimento

```bash
python view_results.py --bucket ppo-flappy-bird --details exp_20250112_143022_abc12345
```

### Descargar el Mejor Checkpoint

```bash
python view_results.py \
    --bucket ppo-flappy-bird \
    --download exp_20250112_143022_abc12345 \
    --output best_model_downloaded.pt
```

### Ver Top Experimentos

```bash
python view_results.py --bucket ppo-flappy-bird --best 5
```

Salida ejemplo:
```
🏆 Top 5 Experiments by Reward:

1. exp_20250112_143022_abc12345
   Reward: 195.50
   Score: 42 pipes
   LR: 3e-04 → 1e-05 (cosine)
   Hidden: 256, Envs: 16

2. exp_20250112_150133_def67890
   Reward: 190.23
   Score: 38 pipes
   ...
```

---

## 📁 Estructura del Bucket

Tu bucket quedará organizado así:

```
gs://ppo-flappy-bird/
│
├── experiments/
│   ├── exp_20250112_143022_abc12345/
│   │   ├── config.json
│   │   ├── checkpoints/
│   │   │   ├── best_model_improved.pt
│   │   │   ├── best_model_improved_full.pt
│   │   │   ├── checkpoint_250000.pt
│   │   │   ├── checkpoint_500000.pt
│   │   │   └── checkpoint_750000.pt
│   │   ├── tensorboard/
│   │   │   └── events.out.tfevents.*
│   │   └── metrics/
│   │       └── final_metrics.json
│   │
│   └── exp_20250112_150133_def67890/
│       └── ...
│
└── search_results/
    └── search_20250112/
        ├── search_config.yaml
        └── results_summary.json
```

---

## 🔄 Recuperación de Experimentos

### Caso 1: Se Perdió la Conexión Durante Entrenamiento

Si tu notebook se desconectó, los checkpoints ya están en GCS. Puedes:

1. **Ver qué se guardó:**
   ```bash
   python view_results.py --bucket ppo-flappy-bird --list
   ```

2. **Descargar el último checkpoint:**
   ```bash
   python view_results.py \
       --bucket ppo-flappy-bird \
       --download exp_XXXXXX \
       --output recovered_model.pt
   ```

3. **Continuar entrenamiento** (feature avanzado - requeriría modificación adicional):
   ```python
   # Cargar checkpoint
   checkpoint = torch.load('recovered_model.pt')
   model.load_state_dict(checkpoint['model'])
   optimizer.load_state_dict(checkpoint['optimizer'])
   start_step = checkpoint['steps']
   ```

### Caso 2: Quiero Re-entrenar con Mejor Config

```bash
# 1. Ver cuál fue la mejor configuración
python view_results.py --bucket ppo-flappy-bird --best 1 --details

# 2. Copiar esos hiperparámetros a config_template.yaml

# 3. Correr con más steps o cambios menores
python run_experiment.py --config config_template.yaml
```

---

## 💡 Tips y Mejores Prácticas

### 1. **Nombrado de Experimentos**

Los experiment IDs se generan automáticamente con el formato:
```
exp_YYYYMMDD_HHMMSS_HASH
```

Donde `HASH` es un hash de la configuración, lo que te permite identificar runs con la misma config.

### 2. **Monitoreo en Colab**

Crea una celda aparte para monitorear:

```python
# Celda 1: Iniciar entrenamiento
!python run_experiment.py --config config_template.yaml

# Celda 2: Monitorear (ejecutar periódicamente)
!python view_results.py --bucket ppo-flappy-bird --compare --top 5
```

### 3. **Upload Asíncrono**

Los uploads a GCS son asíncronos por defecto, así que **no bloquean el entrenamiento**. Los checkpoints se suben en background mientras el entrenamiento continúa.

### 4. **Búsqueda Incremental**

Estrategia recomendada:

1. **Fase 1:** Random search con 10-20 trials
   ```bash
   python run_experiment.py --search --search-config quick_search.yaml
   ```

2. **Fase 2:** Analizar resultados
   ```bash
   python view_results.py --bucket ppo-flappy-bird --best 3
   ```

3. **Fase 3:** Grid search refinado alrededor de los mejores
   - Crea un `refined_search.yaml` con rangos más estrechos
   - Usa grid search con pocas combinaciones

4. **Fase 4:** Entrenamiento largo con el mejor config
   ```bash
   python train_vector_improved.py --total-steps 5000000 --gcs-bucket ppo-flappy-bird ...
   ```

### 5. **Limpieza del Bucket**

Para evitar costos, borra experimentos antiguos/malos:

```python
from gcs_manager import GCSManager

gcs = GCSManager(bucket_name='ppo-flappy-bird', project_id='quiet-sum-477223-g3')

# Listar experimentos
experiments = gcs.list_experiments()

# Borrar uno específico (cuidado!)
# gsutil rm -r gs://ppo-flappy-bird/experiments/exp_XXXXXX/
```

---

## 🐛 Troubleshooting

### Error: "Failed to connect to GCS"

**Solución:**
```python
# En Colab
from google.colab import auth
auth.authenticate_user()

# Verificar
!gcloud auth list
```

### Error: "Permission denied"

Tu cuenta necesita permisos en el bucket. Verifica:
```bash
gsutil ls gs://ppo-flappy-bird
```

### Los archivos no se suben

Verifica que el argumento `--gcs-bucket` esté presente:
```bash
python train_vector_improved.py --gcs-bucket ppo-flappy-bird  # ← importante!
```

### TensorBoard no muestra logs de GCS

Descarga los logs localmente primero:
```bash
gsutil -m cp -r gs://ppo-flappy-bird/experiments/exp_XXXX/tensorboard ./local_tb_logs/
tensorboard --logdir ./local_tb_logs/
```

---

## 📞 Soporte

Si tienes problemas:

1. Verifica que `gcs_manager.py` esté en el mismo directorio
2. Revisa los logs de error completos
3. Verifica la autenticación con `gcloud auth list`
4. Asegúrate que el bucket exista: `gsutil ls gs://ppo-flappy-bird`

---

## 🎯 Ejemplo Completo: Flujo de Trabajo en Colab

```python
# ========================================
# CELDA 1: Setup
# ========================================
!git clone <tu-repo>
%cd TP_FINAL_RL
!pip install -r requirements.txt

from google.colab import auth
auth.authenticate_user()

# ========================================
# CELDA 2: Verificar conexión
# ========================================
from google.cloud import storage
client = storage.Client(project="quiet-sum-477223-g3")
for blob in client.list_blobs("ppo-flappy-bird", max_results=3):
    print(blob.name)

# ========================================
# CELDA 3: Entrenamiento único
# ========================================
!python run_experiment.py --config config_template.yaml

# ========================================
# CELDA 4: Búsqueda de hiperparámetros
# ========================================
!python run_experiment.py \
    --config config_template.yaml \
    --search \
    --search-config search_config_example.yaml

# ========================================
# CELDA 5: Ver resultados (ejecutar después)
# ========================================
!python view_results.py --bucket ppo-flappy-bird --compare --top 10

# ========================================
# CELDA 6: Descargar mejor modelo
# ========================================
!python view_results.py --bucket ppo-flappy-bird --best 1

# Copiar experiment ID del output anterior
!python view_results.py \
    --bucket ppo-flappy-bird \
    --download exp_XXXXXXX_YYYYYY \
    --output best_model.pt
```

---

¡Listo! Ahora tienes un sistema completo para entrenar, buscar hiperparámetros, y gestionar experimentos en GCS sin perder progreso. 🚀
