# play_flappy.py

import gymnasium as gym
import flappy_bird_gymnasium  # importa el registro del env
import pygame


def play_flappy():
    env = gym.make("FlappyBird-v0", render_mode="human")

    obs, info = env.reset(seed=123)

    pygame.init()
    clock = pygame.time.Clock()
    fps = 30  

    running = True
    episode_reward = 0.0
    episode_len = 0
    ep_idx = 0

    while running:
        action = 0  

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    action = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_len += 1

        env.render()

        if terminated or truncated:
            ep_idx += 1
            print(f"Episode {ep_idx} terminado | R = {episode_reward:.1f} | steps = {episode_len}")
            obs, info = env.reset()
            episode_reward = 0.0
            episode_len = 0

        clock.tick(fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    play_flappy()
