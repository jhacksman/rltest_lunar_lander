import math
import random
import numpy as np
from gym import spaces, Env

class LunarLanderEnv(Env):
    """
    Custom Lunar Lander Environment.
    Observation: [x, y, vx, vy, angle, angular_velocity, fuel, altitude]
      - altitude is defined as the distance from the lander's bottom (y + lander_height/2) to the ground.
        When landed, altitude should be 0.
    Actions: Discrete(4): 
      0: nothing, 1: thrust, 2: rotate left, 3: rotate right
    Additional Rule:
      - If the current altitude exceeds 2× the initial altitude (recorded at reset), the episode fails with -1500 points.
    Reward Shaping on Landing:
      - On a successful landing, the reward is higher if the lander’s horizontal position is closer to the center of the landing pad (the green rectangle)
        and if its vertical velocity is low.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LunarLanderEnv, self).__init__()
        self.screen_width = 800
        self.screen_height = 600
        self.flat_ground = 550  # The top of the landing pad (green rectangle) is here.
        self.gravity = 0.05
        self.thrust_power = 0.2
        self.rotation_speed = math.radians(5)
        self.fuel_consumption = 0.5

        # Lander dimensions
        self.lander_width = 20
        self.lander_height = 30
        self.leg_length = 10

        # Observation: [x, y, vx, vy, angle, angular_velocity, fuel, altitude]
        # Altitude is measured from the lander's bottom (y + lander_height/2) to the ground.
        low = np.array([0, 0, -np.inf, -np.inf, -math.pi, -np.inf, 0, 0], dtype=np.float32)
        high = np.array([self.screen_width, self.screen_height, np.inf, np.inf,
                         math.pi, np.inf, 100, self.screen_height], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self):
        self.x = random.uniform(self.screen_width * 0.25, self.screen_width * 0.75)
        self.y = 50
        self.vx = 0
        self.vy = 0
        self.angle = 0
        self.angular_velocity = 0
        self.fuel = 100
        self.done = False
        self.reward = 0
        # Record the initial altitude: distance from the lander's bottom to the ground.
        self.initial_altitude = self._get_ground_height(self.x) - (self.y + self.lander_height / 2)
        return self._get_obs()

    def _get_ground_height(self, x):
        if x < 0.1 * self.screen_width:
            max_height = self.flat_ground + (0.1 * self.screen_width) * math.tan(math.radians(30))
            ground = max_height - (x * math.tan(math.radians(30)))
        elif x > 0.9 * self.screen_width:
            ground = self.flat_ground + ((x - 0.9 * self.screen_width) * math.tan(math.radians(30)))
        else:
            ground = self.flat_ground
        return ground

    def _rotate_point(self, point):
        """Rotate a local point by self.angle and translate by (self.x, self.y)."""
        px, py = point
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        return (self.x + rx, self.y + ry)

    def _get_obs(self):
        # Altitude is the distance from the lander's bottom to the ground.
        altitude = self._get_ground_height(self.x) - (self.y + self.lander_height / 2)
        base_obs = np.array([self.x, self.y, self.vx, self.vy, self.angle, self.angular_velocity, self.fuel], dtype=np.float32)
        return np.concatenate((base_obs, [altitude]), axis=0)

    def step(self, action):
        if self.fuel > 0:
            if action == 1:  # thrust
                thrust = self.thrust_power
                self.fuel = max(self.fuel - self.fuel_consumption, 0)
            else:
                thrust = 0
            if action == 2:  # rotate left
                self.angle -= self.rotation_speed
            elif action == 3:  # rotate right
                self.angle += self.rotation_speed
        else:
            thrust = 0

        thrust_x = thrust * math.sin(self.angle)
        thrust_y = -thrust * math.cos(self.angle)
        self.vx += thrust_x
        self.vy += thrust_y + self.gravity
        self.x += self.vx
        self.y += self.vy

        # Compute current altitude from lander's bottom to ground.
        current_altitude = self._get_ground_height(self.x) - (self.y + self.lander_height / 2)

        # Altitude failure: if current altitude > 2 * initial altitude, it's a severe failure.
        if current_altitude > 2 * self.initial_altitude:
            self.done = True
            self.reward = -1500
            return self._get_obs(), self.reward, self.done, {"altitude_failure": True}

        # Reward shaping based on vertical velocity and altitude.
        # Define target vertical speed thresholds based on altitude ranges:
        # - If altitude > 200: no limit.
        # - 200 >= altitude > 100: target threshold = 5.
        # - 100 >= altitude > 50: target threshold = 4.
        # - altitude <= 50: target threshold = 3.
        if current_altitude > 200:
            target_threshold = float('inf')
        elif current_altitude > 100:
            target_threshold = 5
        elif current_altitude > 50:
            target_threshold = 4
        else:
            target_threshold = 3

        if self.vy > target_threshold:
            shaping_reward = -(self.vy - target_threshold)
        else:
            shaping_reward = 0

        step_reward = -0.1 + shaping_reward

        # Collision detection using body and legs.
        body_points = [(0, -self.lander_height/2),
                       (self.lander_width/2, self.lander_height/2),
                       (-self.lander_width/2, self.lander_height/2)]
        body_vertices = [self._rotate_point(p) for p in body_points]
        body_bottom = max(y for (_, y) in body_vertices)

        left_leg_end = self._rotate_point((-self.lander_width/2, self.lander_height/2 + self.leg_length))
        right_leg_end = self._rotate_point((self.lander_width/2, self.lander_height/2 + self.leg_length))

        ground_body = self._get_ground_height(self.x)
        ground_left = self._get_ground_height(left_leg_end[0])
        ground_right = self._get_ground_height(right_leg_end[0])

        body_contact = (body_bottom >= ground_body)
        legs_contact = (left_leg_end[1] >= ground_left) and (right_leg_end[1] >= ground_right)

        if body_contact or legs_contact:
            self.done = True
            landing_success = False
            body_margin = body_bottom - ground_body
            if legs_contact and (body_margin <= 5) and (abs(self.vy) < 3) and (abs(self.vx) < 3) and (abs(self.angle) < math.radians(15)):
                landing_success = True
            if landing_success:
                center_x = self.screen_width / 2
                # Incorporate horizontal proximity reward.
                horizontal_penalty = abs(self.x - center_x) * 0.5  # adjust factor as needed
                velocity_penalty = abs(self.vx) + abs(self.vy)
                reward = 200 - (horizontal_penalty + velocity_penalty)
                # Set y so that the lander's bottom touches the ground (altitude = 0).
                self.y = self._get_ground_height(self.x) - self.lander_height/2
            else:
                reward = -100
            self.reward = reward
            return self._get_obs(), reward, self.done, {"landing_success": landing_success}
        
        self.reward = step_reward
        return self._get_obs(), step_reward, self.done, {}

    def render(self, mode='human'):
        import pygame  # local import
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((0, 0, 0))
        terrain_points = [(0, self._get_ground_height(0))]
        for x in range(0, self.screen_width + 1, 20):
            terrain_points.append((x, self._get_ground_height(x)))
        terrain_points += [(self.screen_width, self._get_ground_height(self.screen_width)),
                           (self.screen_width, self.screen_height),
                           (0, self.screen_height)]
        pygame.draw.polygon(surface, (100, 100, 100), terrain_points)
        marker_x = self.screen_width // 2
        pygame.draw.rect(surface, (0, 255, 0), (marker_x - 20, self.flat_ground, 40, 5))
        body_points = [(0, -self.lander_height/2),
                       (self.lander_width/2, self.lander_height/2),
                       (-self.lander_width/2, self.lander_height/2)]
        body_vertices = [self._rotate_point(p) for p in body_points]
        pygame.draw.polygon(surface, (255, 255, 255), body_vertices)
        left_leg_start = self._rotate_point((-self.lander_width/2, self.lander_height/2))
        left_leg_end = self._rotate_point((-self.lander_width/2, self.lander_height/2 + self.leg_length))
        right_leg_start = self._rotate_point((self.lander_width/2, self.lander_height/2))
        right_leg_end = self._rotate_point((self.lander_width/2, self.lander_height/2 + self.leg_length))
        pygame.draw.line(surface, (200, 200, 200), left_leg_start, left_leg_end, 3)
        pygame.draw.line(surface, (200, 200, 200), right_leg_start, right_leg_end, 3)
        fuel_bar_length = 50
        fuel_ratio = self.fuel / 100
        pygame.draw.rect(surface, (255, 0, 0), (10, 10, fuel_bar_length, 10))
        pygame.draw.rect(surface, (0, 255, 0), (10, 10, fuel_bar_length * fuel_ratio, 10))
        return surface

