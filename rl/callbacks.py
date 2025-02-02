from stable_baselines3.common.callbacks import BaseCallback
import globals

class TrainingProgressCallback(BaseCallback):
    """
    A custom callback that updates global training_metrics.
    """
    def __init__(self, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env
        episode_rewards = getattr(env, "episode_rewards", [])
        globals.training_metrics["episode_count"] = len(episode_rewards)
        globals.training_metrics["episode_rewards"] = episode_rewards
        episode_successes = getattr(env, "episode_successes", [])
        if episode_successes:
            globals.training_metrics["success_rate"] = sum(episode_successes) / len(episode_successes)
        else:
            globals.training_metrics["success_rate"] = 0
        return True

