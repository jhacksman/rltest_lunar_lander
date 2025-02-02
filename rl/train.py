import os
import time
from stable_baselines3 import PPO
from rl.callbacks import TrainingProgressCallback
from env.lunar_lander import LunarLanderEnv
from env.monitor import Monitor
import globals

MODEL_FILENAME = "lunar_lander_model.zip"

def run_training(config, training_metrics, batch_start_episode_ref):
    """
    Train for a batch of timesteps. If a model and environment already exist,
    continue training from where they left off. Otherwise, load from disk if available.
    """
    # Create a persistent training environment if needed.
    if globals.train_env is None:
        globals.train_env = Monitor(LunarLanderEnv())
        batch_start_episode_ref[0] = 0
    else:
        env = globals.train_env.envs[0] if hasattr(globals.train_env, "envs") else globals.train_env
        batch_start_episode_ref[0] = len(env.episode_rewards)
    
    # Check if a saved model exists on disk, and load it if globals.model is not set.
    if globals.model is None and os.path.exists(MODEL_FILENAME):
        globals.model = PPO.load(MODEL_FILENAME, env=globals.train_env)
        print("Loaded saved model from disk.")
    elif globals.model is None:
        globals.model = PPO("MlpPolicy", globals.train_env, learning_rate=config["learning_rate"],
                              batch_size=config["batch_size"], verbose=0)
    
    progress_callback = TrainingProgressCallback()
    total_timesteps = config["num_episodes"] * 1000
    globals.model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    
    env = globals.train_env.envs[0] if hasattr(globals.train_env, "envs") else globals.train_env
    training_metrics["episode_rewards"] = env.episode_rewards
    training_metrics["episode_count"] = len(env.episode_rewards)
    if env.episode_successes:
        training_metrics["success_rate"] = sum(env.episode_successes) / len(env.episode_successes)
    else:
        training_metrics["success_rate"] = 0

def save_model(MODEL_FILENAME):
    if globals.model is not None:
        globals.model.save(MODEL_FILENAME)
        print("Model saved to", MODEL_FILENAME)

