"""
🔍 DIAGNÓSTICO 1: Verificar qué observaciones estás recibiendo
Corre esto ANTES de entrenar para confirmar vector vs píxeles
"""
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

print("=" * 70)
print("🔍 DIAGNÓSTICO: Observaciones de FlappyBird-v0")
print("=" * 70)

env = gym.make("FlappyBird-v0")
obs, info = env.reset()

print(f"\n📊 Tipo: {type(obs)}")
print(f"📐 Shape: {obs.shape}")
print(f"📏 Dtype: {obs.dtype}")
print(f"📈 Min: {obs.min():.3f}, Max: {obs.max():.3f}")
print(f"📦 Tamaño total: {obs.size} elementos")

if obs.ndim == 1:
    print(f"\n✅ SON VECTORES (1D)")
    print(f"   Dimensión: {obs.shape[0]} features")
    print(f"\n🔢 Valores de ejemplo:")
    for i, val in enumerate(obs):
        print(f"   Feature {i:2d}: {val:8.3f}")
    
    print(f"\n💡 Estadísticas después de 100 pasos:")
    all_obs = []
    for _ in range(100):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
        all_obs.append(obs)
    
    all_obs = np.array(all_obs)
    print(f"   Means: {all_obs.mean(axis=0)}")
    print(f"   Stds:  {all_obs.std(axis=0)}")
    print(f"   Mins:  {all_obs.min(axis=0)}")
    print(f"   Maxs:  {all_obs.max(axis=0)}")
    
    # Verificar si necesitan normalización
    max_range = (all_obs.max(axis=0) - all_obs.min(axis=0)).max()
    if max_range > 10:
        print(f"\n⚠️  ALERTA: Rango máximo = {max_range:.1f}")
        print(f"   📝 RECOMENDACIÓN: Normalizar features")
        print(f"      - Opción 1: Dividir por constantes (ancho/alto pantalla)")
        print(f"      - Opción 2: Estandarizar online (mean=0, std=1)")
    else:
        print(f"\n✅ Rangos razonables (max={max_range:.1f}), pero aún considera normalizar")

elif obs.ndim == 3:
    print(f"\n📺 SON PÍXELES (HxWxC o CxHxW)")
    h, w, c = obs.shape if obs.shape[2] <= 4 else (obs.shape[1], obs.shape[2], obs.shape[0])
    print(f"   Dimensiones: H={h}, W={w}, Canales={c}")
    
    if obs.max() > 1.0:
        print(f"\n⚠️  ALERTA: Valores sin normalizar (max={obs.max()})")
        print(f"   📝 RECOMENDACIÓN: Dividir por 255.0")
    
    if c == 3:
        print(f"\n💡 SUGERENCIA: Convertir a escala de grises")
        print(f"   - Ahorrás 3x en memoria/compute")
    
    print(f"\n🎯 Resoluciones a probar (en orden):")
    for size in [48, 64, 72, 84]:
        pixels = size * size * (1 if c == 1 else c)
        print(f"   {size}x{size}: {pixels:,} píxeles → CNN necesaria")

else:
    print(f"\n❓ Formato desconocido: {obs.shape}")

env.close()
print("\n" + "=" * 70)
