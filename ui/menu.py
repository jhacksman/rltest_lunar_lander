import pygame
import sys
import time
import threading
import os
from ui import widgets
from rl import train
from ui import demo
import globals  # Shared globals: training_metrics, model, train_env, etc.

# Rectangle for the on-screen "Save & Continue" button in training mode.
SAVE_CONTINUE_RECT = pygame.Rect(50, 250, 200, 40)
# Rectangle for the on-screen "Save & Exit" button in training mode.
SAVE_EXIT_RECT = pygame.Rect(50, 310, 200, 40)

def main_menu():
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Lunar Lander RL Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Create input fields.
    sim_speed_input = widgets.TextInput((300, 150, 100, 30), "1", "Set simulation speed (1x-100x)")
    episodes_input = widgets.TextInput((300, 200, 100, 30), "100", "Training episodes per batch")
    lr_input = widgets.TextInput((300, 250, 100, 30), "0.0003", "Learning rate")
    batch_input = widgets.TextInput((300, 300, 100, 30), "64", "Batch size")
    inputs = [sim_speed_input, episodes_input, lr_input, batch_input]

    buttons = []
    def start_training():
        config = {
            "simulation_speed": int(sim_speed_input.text),
            "num_episodes": int(episodes_input.text),
            "learning_rate": float(lr_input.text),
            "batch_size": int(batch_input.text)
        }
        training_mode(screen, font, config)

    def start_demo():
        demo.demo_mode(screen, font)

    def save_model():
        train.save_model("lunar_lander_model.zip")

    def load_model():
        from stable_baselines3 import PPO
        try:
            global model
            globals.model = PPO.load("lunar_lander_model.zip")
            print("Model loaded.")
        except Exception as e:
            print("Error loading model:", e)

    buttons.append(widgets.Button((150, 400, 150, 40), "Training Mode", start_training, "Enter training mode"))
    buttons.append(widgets.Button((350, 400, 150, 40), "Demo Mode", start_demo, "Enter demo mode"))
    buttons.append(widgets.Button((150, 450, 150, 40), "Save Model", save_model, "Save current model"))
    buttons.append(widgets.Button((350, 450, 150, 40), "Load Model", load_model, "Load saved model"))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for inp in inputs:
                inp.handle_event(event)
            for btn in buttons:
                btn.handle_event(event)
        screen.fill((50, 50, 80))
        title = font.render("Lunar Lander RL Simulation", True, (255, 255, 255))
        screen.blit(title, (250, 50))
        label_font = pygame.font.SysFont(None, 20)
        screen.blit(label_font.render("Sim Speed:", True, (255,255,255)), (200,155))
        screen.blit(label_font.render("Episodes:", True, (255,255,255)), (200,205))
        screen.blit(label_font.render("Learning Rate:", True, (255,255,255)), (200,255))
        screen.blit(label_font.render("Batch Size:", True, (255,255,255)), (200,305))
        for inp in inputs:
            inp.draw(screen, font)
        for btn in buttons:
            btn.draw(screen, font)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    sys.exit()

def training_mode(screen, font, config):
    global globals
    # batch_start_episode_ref holds the episode count at the start of this training batch.
    batch_start_episode_ref = [0]
    training_thread = threading.Thread(
        target=train.run_training, 
        args=(config, globals.training_metrics, batch_start_episode_ref)
    )
    training_thread.start()

    running = True
    clock = pygame.time.Clock()
    last_plot_update_time = time.time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # When training batch is complete, check for clicks on "Save & Continue" and "Save & Exit"
            if not training_thread.is_alive():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if SAVE_CONTINUE_RECT.collidepoint(event.pos):
                        # Save model and continue training
                        train.save_model("lunar_lander_model.zip")
                        from stable_baselines3 import PPO
                        globals.model = PPO.load("lunar_lander_model.zip", env=globals.train_env)
                        training_thread = threading.Thread(
                            target=train.run_training, 
                            args=(config, globals.training_metrics, batch_start_episode_ref)
                        )
                        training_thread.start()
                    elif SAVE_EXIT_RECT.collidepoint(event.pos):
                        # Save model and exit training mode to return to main menu.
                        train.save_model("lunar_lander_model.zip")
                        running = False
        screen.fill((0, 0, 0))
        current_episodes = globals.training_metrics.get("episode_count", 0)
        batch_progress = current_episodes - batch_start_episode_ref[0]
        batch_progress = max(batch_progress, 0)
        progress_percentage = min(batch_progress / config["num_episodes"], 1.0) * 100
        lines = [
            f"Total Episodes Trained: {current_episodes}",
            f"Batch Progress: {batch_progress} / {config['num_episodes']} episodes ({progress_percentage:.1f}%)",
            f"Success Rate: {globals.training_metrics.get('success_rate',0)*100:.1f}%",
            f"Last Episode Reward: {globals.training_metrics['episode_rewards'][-1] if globals.training_metrics.get('episode_rewards') else 'N/A'}"
        ]
        for i, line in enumerate(lines):
            text = font.render(line, True, (255,255,255))
            screen.blit(text, (50, 50 + i * 30))
        if not training_thread.is_alive():
            # Draw "Save & Continue" button.
            pygame.draw.rect(screen, (0, 100, 0), SAVE_CONTINUE_RECT)
            sc_text = font.render("Save & Continue", True, (255,255,255))
            sc_rect = sc_text.get_rect(center=SAVE_CONTINUE_RECT.center)
            screen.blit(sc_text, sc_rect)
            # Draw "Save & Exit" button below it.
            pygame.draw.rect(screen, (100, 0, 0), SAVE_EXIT_RECT)
            se_text = font.render("Save & Exit", True, (255,255,255))
            se_rect = se_text.get_rect(center=SAVE_EXIT_RECT.center)
            screen.blit(se_text, se_rect)
        if time.time() - last_plot_update_time > 10:
            last_plot_update_time = time.time()
            from utils.plot_utils import update_plot
            if globals.training_metrics.get("episode_rewards"):
                update_plot(globals.training_metrics["episode_rewards"])
        if os.path.exists("temp_plot.png"):
            try:
                plot_img = pygame.image.load("temp_plot.png")
                plot_img = pygame.transform.smoothscale(plot_img, (300, 200))
                screen.blit(plot_img, (500, 0))
            except Exception as e:
                print("Error loading plot image:", e)
        pygame.display.flip()
        clock.tick(60)
    # After exiting training mode, return to main menu.
    main_menu()

