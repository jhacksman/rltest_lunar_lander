import pygame
import time
from env.lunar_lander import LunarLanderEnv
import globals  # Use shared globals for the model

def demo_mode(screen, font):
    demo_env = LunarLanderEnv()
    demo_manual = False
    demo_toggle_rect = pygame.Rect(700, 10, 90, 30)
    demo_toggle_text = "AI Mode"
    clock = pygame.time.Clock()
    last_step_time = time.time()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Check for manual override using WASD or arrow keys
                if event.key in [pygame.K_w, pygame.K_UP]:
                    demo_manual = True
                    demo_toggle_text = "Manual"
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    demo_manual = True
                    demo_toggle_text = "Manual"
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    demo_manual = True
                    demo_toggle_text = "Manual"
                elif event.key == pygame.K_ESCAPE:
                    # Exit demo mode and return to main menu
                    running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if demo_toggle_rect.collidepoint(event.pos):
                    demo_manual = False
                    demo_toggle_text = "AI Mode"
        now = time.time()
        if now - last_step_time > 1/30:
            last_step_time = now
            if not demo_manual and globals.model is not None:
                obs = demo_env._get_obs()
                action, _ = globals.model.predict(obs, deterministic=True)
            else:
                keys = pygame.key.get_pressed()
                # Check both WASD and arrow keys:
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action = 1
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = 2
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    action = 3
                else:
                    action = 0
            obs, reward, done, info = demo_env.step(action)
            if done:
                # Display win/lose message based on landing success.
                if info.get("landing_success", False):
                    result_text = "WIN"
                    result_color = (0, 255, 0)
                else:
                    result_text = "CRASH"
                    result_color = (255, 0, 0)
                result_surf = font.render(result_text, True, result_color)
                screen.blit(result_surf, (int(demo_env.x), int(demo_env.y)))
                pygame.display.flip()
                pygame.time.wait(1000)
                demo_env.reset()
                demo_manual = False
                demo_toggle_text = "AI Mode"
        env_surface = demo_env.render()
        screen.blit(env_surface, (0, 0))
        # Get altitude from observation (last element)
        obs = demo_env._get_obs()
        altitude = obs[-1]
        telemetry = [
            f"Velocity: ({demo_env.vx:.1f}, {demo_env.vy:.1f})",
            f"Angle: {round(demo_env.angle * 180/3.1416,1)}°",
            f"Fuel: {demo_env.fuel:.1f}",
            f"Altitude: {altitude:.1f}",
            f"Reward: {demo_env.reward:.1f}"
        ]
        for i, line in enumerate(telemetry):
            t = font.render(line, True, (255, 255, 0))
            screen.blit(t, (10, 10 + i * 20))
        pygame.draw.rect(screen, (100,100,255), demo_toggle_rect)
        t_toggle = font.render(demo_toggle_text, True, (255,255,255))
        screen.blit(t_toggle, demo_toggle_rect)
        menu_text = font.render("Press Esc to return to menu.", True, (255,255,255))
        screen.blit(menu_text, (600,570))
        pygame.display.flip()
        clock.tick(60)

