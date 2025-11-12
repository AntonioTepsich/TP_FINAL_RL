"""
Ejemplo de uso en Google Colab
Este script muestra cómo configurar y ejecutar experimentos desde Colab
Copia este código en celdas de Colab según se indica
"""

# ============================================================================
# CELDA 1: Setup e Instalación
# ============================================================================
"""
# Clonar repositorio (si está en GitHub)
!git clone https://github.com/tu-usuario/TP_FINAL_RL.git
%cd TP_FINAL_RL

# O si trabajas con archivos locales, súbelos a Colab
from google.colab import files
# uploaded = files.upload()

# Instalar dependencias
!pip install -q -r requirements.txt

print("✅ Setup completado")
"""

# ============================================================================
# CELDA 2: Autenticación con GCP
# ============================================================================
"""
from google.colab import auth
auth.authenticate_user()

# Verificar autenticación
!gcloud auth list

print("✅ Autenticación completada")
"""

# ============================================================================
# CELDA 3: Verificar Conexión al Bucket
# ============================================================================
"""
from google.cloud import storage

PROJECT_ID = "quiet-sum-477223-g3"
BUCKET = "ppo-flappy-bird"

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET)

# Listar archivos existentes
print(f"📦 Contenido de gs://{BUCKET}:")
blobs = list(client.list_blobs(BUCKET, max_results=10))

if blobs:
    for blob in blobs:
        print(f"  - {blob.name} ({blob.size / 1024:.1f} KB)")
else:
    print("  (bucket vacío)")

print(f"\\n✅ Conexión exitosa a gs://{BUCKET}")
"""

# ============================================================================
# CELDA 4: Configuración Rápida (Opcional)
# ============================================================================
"""
# Si quieres modificar la config desde Colab sin editar el YAML

import yaml

config = {
    'n_envs': 16,
    'rollout_steps': 256,
    'total_steps': 500000,  # Reducido para prueba rápida
    'hidden_size': 256,
    'lr_start': 0.0003,
    'lr_end': 0.00001,
    'lr_schedule': 'cosine',
    'ent_start': 0.02,
    'ent_end': 0.005,
    'ent_schedule': 'linear',
    'clip_epsilon': 0.2,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'epochs_per_update': 4,
    'minibatch_size': 1024,
    'normalize_obs': True,
    'gcs_bucket': 'ppo-flappy-bird',
    'gcs_project': 'quiet-sum-477223-g3',
    'experiment_id': None,
    'upload_interval': 250000,
    'log_dir': 'runs',
    'comment': 'colab_experiment',
    'checkpoint_interval': 250000,
    'seed': 42,
    'device': None
}

# Guardar config personalizada
with open('my_colab_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Configuración personalizada guardada en my_colab_config.yaml")
"""

# ============================================================================
# CELDA 5A: Entrenamiento Simple (Opción 1)
# ============================================================================
"""
# Ejecutar con configuración por defecto
!python run_experiment.py --config config_template.yaml

# O con config personalizada
# !python run_experiment.py --config my_colab_config.yaml
"""

# ============================================================================
# CELDA 5B: Entrenamiento con Parámetros Custom (Opción 2)
# ============================================================================
"""
# Ejecutar directamente con argumentos CLI
!python train_vector_improved.py \\
    --gcs-bucket ppo-flappy-bird \\
    --gcs-project quiet-sum-477223-g3 \\
    --total-steps 500000 \\
    --lr-start 0.0003 \\
    --lr-end 0.00001 \\
    --hidden-size 256 \\
    --n-envs 16 \\
    --comment "colab_test_run" \\
    --seed 42
"""

# ============================================================================
# CELDA 6: Búsqueda de Hiperparámetros (OPCIONAL)
# ============================================================================
"""
# Primero, crear una configuración de búsqueda rápida
search_config = {
    'strategy': 'random',
    'n_trials': 5,  # Solo 5 trials para prueba rápida
    'seed': 42,
    'search_space': {
        'lr_start': [0.0001, 0.0003, 0.0005],
        'hidden_size': [128, 256, 512],
        'clip_epsilon': [0.1, 0.2, 0.3],
        'n_envs': [8, 16]
    }
}

import yaml
with open('quick_search.yaml', 'w') as f:
    yaml.dump(search_config, f)

# Ejecutar búsqueda
!python run_experiment.py \\
    --config config_template.yaml \\
    --search \\
    --search-config quick_search.yaml \\
    --output-dir search_results

print("\\n🔍 Búsqueda de hiperparámetros completada")
"""

# ============================================================================
# CELDA 7: Monitorear Progreso Durante Entrenamiento
# ============================================================================
"""
# Ejecutar esta celda periódicamente mientras entrena
# (en otra pestaña o después de iniciar el entrenamiento en background)

import time
from IPython.display import clear_output

def monitor_experiments(bucket_name='ppo-flappy-bird', refresh_interval=30):
    '''Monitorea experimentos en tiempo real'''
    try:
        while True:
            clear_output(wait=True)
            print(f"🔄 Actualizando... ({time.strftime('%H:%M:%S')})")
            print("=" * 80)

            # Listar y comparar experimentos
            !python view_results.py --bucket {bucket_name} --compare --top 5 --format simple

            print(f"\\n⏰ Próxima actualización en {refresh_interval}s...")
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\\n⏸️  Monitoreo detenido")

# Usar así (va a refrescar cada 30 segundos):
# monitor_experiments()

# O simplemente ver el estado actual una vez:
!python view_results.py --bucket ppo-flappy-bird --compare --top 5
"""

# ============================================================================
# CELDA 8: Ver Resultados Finales
# ============================================================================
"""
# Listar todos los experimentos
!python view_results.py --bucket ppo-flappy-bird --list

print("\\n" + "="*80 + "\\n")

# Ver top 5 mejores
!python view_results.py --bucket ppo-flappy-bird --best 5

print("\\n" + "="*80 + "\\n")

# Tabla comparativa completa
!python view_results.py --bucket ppo-flappy-bird --compare --top 10 --format fancy_grid

# Exportar a CSV
!python view_results.py --bucket ppo-flappy-bird --compare --export-csv --output results.csv

print("\\n✅ Resultados guardados en results.csv")
"""

# ============================================================================
# CELDA 9: Ver Detalles de un Experimento Específico
# ============================================================================
"""
# Primero, obtén el experiment_id de la celda anterior
# Reemplaza EXPERIMENT_ID con el ID real

EXPERIMENT_ID = "exp_20250112_143022_abc12345"  # ← Cambiar esto

!python view_results.py --bucket ppo-flappy-bird --details {EXPERIMENT_ID}
"""

# ============================================================================
# CELDA 10: Descargar Mejor Modelo
# ============================================================================
"""
# Opción 1: Descargar modelo específico
EXPERIMENT_ID = "exp_20250112_143022_abc12345"  # ← Cambiar esto

!python view_results.py \\
    --bucket ppo-flappy-bird \\
    --download {EXPERIMENT_ID} \\
    --output best_model_downloaded.pt

print(f"\\n✅ Modelo descargado: best_model_downloaded.pt")

# Opción 2: Descargar directamente con gsutil (más rápido)
!gsutil cp gs://ppo-flappy-bird/experiments/{EXPERIMENT_ID}/checkpoints/best_model_improved_full.pt ./

# Verificar descarga
import os
if os.path.exists('best_model_downloaded.pt'):
    size_mb = os.path.getsize('best_model_downloaded.pt') / (1024 * 1024)
    print(f"✅ Archivo descargado: {size_mb:.2f} MB")
"""

# ============================================================================
# CELDA 11: Cargar y Evaluar Modelo Descargado
# ============================================================================
"""
import torch
import gymnasium as gym
import numpy as np

# Cargar el modelo
checkpoint = torch.load('best_model_downloaded.pt', map_location='cpu')

print("📦 Checkpoint contiene:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Recrear el modelo
from train_vector_improved import VectorActorCritic

model = VectorActorCritic(obs_dim=180, n_actions=2, hidden=256)
model.load_state_dict(checkpoint['model'])
model.eval()

print(f"\\n✅ Modelo cargado")
print(f"   Best Reward: {checkpoint.get('best_reward', 'N/A')}")
print(f"   Steps: {checkpoint.get('global_steps', 'N/A'):,}")

# Opcional: Evaluar el modelo
import flappy_bird_gymnasium

env = gym.make('FlappyBird-v0')

def evaluate_model(model, env, n_episodes=5):
    scores = []
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t)
                action = logits.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        score = info.get('score', 0)
        scores.append(score)
        rewards.append(episode_reward)
        print(f"Episode {ep+1}: Score={score}, Reward={episode_reward:.1f}")

    print(f"\\nPromedio: Score={np.mean(scores):.1f}, Reward={np.mean(rewards):.1f}")
    return scores, rewards

scores, rewards = evaluate_model(model, env, n_episodes=10)

env.close()
"""

# ============================================================================
# CELDA 12: Visualizar TensorBoard (OPCIONAL)
# ============================================================================
"""
# Descargar logs de TensorBoard
EXPERIMENT_ID = "exp_20250112_143022_abc12345"  # ← Cambiar esto

!gsutil -m cp -r gs://ppo-flappy-bird/experiments/{EXPERIMENT_ID}/tensorboard ./tb_logs/

# Iniciar TensorBoard en Colab
%load_ext tensorboard
%tensorboard --logdir ./tb_logs/

print("\\n✅ TensorBoard iniciado")
print("   Abre el enlace que aparece arriba ⬆️")
"""

# ============================================================================
# CELDA 13: Limpiar Archivos Locales (Opcional)
# ============================================================================
"""
# Si necesitas liberar espacio en Colab

import shutil
import os

# Borrar logs de TensorBoard locales
if os.path.exists('runs'):
    shutil.rmtree('runs')
    print("✅ Borrado: runs/")

# Borrar checkpoints locales (ya están en GCS)
for file in os.listdir('.'):
    if file.endswith('.pt'):
        os.remove(file)
        print(f"✅ Borrado: {file}")

# Borrar cache de GCS
if os.path.exists('gcs_cache'):
    shutil.rmtree('gcs_cache')
    print("✅ Borrado: gcs_cache/")

print("\\n🧹 Limpieza completada")
"""

# ============================================================================
# CELDA 14: Continuar Entrenamiento (AVANZADO)
# ============================================================================
"""
# Si quieres continuar desde un checkpoint existente

# 1. Descargar checkpoint
EXPERIMENT_ID = "exp_20250112_143022_abc12345"
!gsutil cp gs://ppo-flappy-bird/experiments/{EXPERIMENT_ID}/checkpoints/checkpoint_750000.pt ./

# 2. Cargar checkpoint
import torch
checkpoint = torch.load('checkpoint_750000.pt')

print(f"Checkpoint en step: {checkpoint['steps']:,}")
print(f"LR actual: {checkpoint['lr']:.2e}")
print(f"Entropy coef: {checkpoint['ent_coef']:.4f}")

# 3. Modificar train_vector_improved.py para cargar el checkpoint
# (esto requeriría agregar lógica de resume al script)

# Por ahora, puedes usar el modelo cargado para continuar manualmente
# o simplemente iniciar un nuevo experimento con los mejores hiperparámetros
"""

# ============================================================================
# TIPS Y NOTAS
# ============================================================================
"""
TIPS IMPORTANTES:

1. **Mantén el Notebook Activo:**
   - Colab puede desconectar después de inactividad
   - Los uploads a GCS son automáticos, así que no perderás progreso
   - Puedes ver resultados parciales en cualquier momento

2. **Monitoreo:**
   - Ejecuta la celda de monitoreo en paralelo
   - Los checkpoints se suben cada 250k steps por defecto

3. **Memoria:**
   - Si Colab se queda sin RAM, reduce n_envs
   - Reduce batch_size si es necesario

4. **GPU:**
   - Asegúrate de tener GPU habilitada: Runtime → Change runtime type → GPU
   - Verifica con: !nvidia-smi

5. **Búsqueda de Hiperparámetros:**
   - Empieza con pocas trials (5-10)
   - Usa random search primero
   - Grid search solo para espacios muy pequeños

6. **Costos de GCS:**
   - Storage es barato (~$0.02/GB/mes)
   - Pero borra experimentos fallidos/viejos

7. **Debugging:**
   - Si algo falla, revisa los logs completos
   - Verifica autenticación: !gcloud auth list
   - Verifica bucket: !gsutil ls gs://ppo-flappy-bird

ATAJOS ÚTILES:

# Ver solo los mejores 3 experimentos
!python view_results.py --bucket ppo-flappy-bird --best 3

# Descargar el mejor modelo automáticamente
# (primero identifica cuál es el mejor con --best 1)
!python view_results.py --bucket ppo-flappy-bird --best 1 | grep "experiment_id"

# Ver todos los archivos de un experimento
!gsutil ls -r gs://ppo-flappy-bird/experiments/exp_XXXXX/

# Descargar TODO un experimento
!gsutil -m cp -r gs://ppo-flappy-bird/experiments/exp_XXXXX/ ./
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Este archivo contiene ejemplos de código para usar en Google Colab")
    print("Copia cada sección en una celda separada de Colab")
    print("="*80)
