# play_flappy.py

import gymnasium as gym
import flappy_bird_gymnasium  # importa el registro del env
import pygame


def play_flappy():
    # Si querés vector obs en lugar de píxeles:
    # env = gym.make("FlappyBird-v0", use_lidar=True, render_mode="human")
    env = gym.make("FlappyBird-v0", render_mode="human")

    obs, info = env.reset(seed=123)

    # Pygame se inicializa internamente, pero por si acaso:
    pygame.init()
    clock = pygame.time.Clock()
    fps = 30  # frames por segundo aprox

    running = True
    episode_reward = 0.0
    episode_len = 0
    ep_idx = 0

    print("Controles:")
    print("  SPACE o ↑  = aletear")
    print("  ESC / Q    = salir")

    while running:
        action = 0  # 0 = no aletear, 1 = aletear

        # Procesar eventos de pygame (cerrar ventana, teclas, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1

        # También podés chequear teclas “sostenidas”
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        # Un paso en el env
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_len += 1

        # Render
        env.render()

        # Si termina el episodio (choque o límite de tiempo)
        if terminated or truncated:
            ep_idx += 1
            print(f"Episode {ep_idx} terminado | R = {episode_reward:.1f} | steps = {episode_len}")
            obs, info = env.reset()
            episode_reward = 0.0
            episode_len = 0

        # Limitamos FPS para que se pueda jugar
        clock.tick(fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    play_flappy()
