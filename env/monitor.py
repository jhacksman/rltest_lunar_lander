from gym import Wrapper

class Monitor(Wrapper):
    """
    A wrapper that records episode rewards, lengths, and successes.
    """
    def __init__(self, env):
        super(Monitor, self).__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_reward = 0
        self.current_length = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.current_length > 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_successes.append(1 if self.current_reward > -50 else 0)
        self.current_reward = 0
        self.current_length = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_reward += reward
        self.current_length += 1
        return obs, reward, done, info

