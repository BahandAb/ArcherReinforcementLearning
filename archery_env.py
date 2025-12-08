import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class ArcheryGymEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(ArcheryGymEnv, self).__init__()
        
        # Pygame Setup
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Physics Constants
        self.gravity = 0.5
        self.target_radius = 20

        # --- FIX 1: Normalized Action Space [-1, 1] ---
        # This fixes the "UserWarning" and makes training faster.
        # We use [-1, 1] because Neural Networks learn better with these numbers.
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )

        # --- FIX 2: Infinite Observation Space ---
        # This fixes the "AssertionError" (Crash) when the arrow flies off-screen.
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )

        # Pygame Setup
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Physics Constants
        self.gravity = 0.5
        self.target_radius = 20

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed) # Seed the internal RNG

        # Randomize Target
        self.target_pos = np.array([
            self.np_random.integers(400, self.width - 50),
            self.np_random.integers(100, self.height - 100)
        ], dtype=np.float32)

        # Reset Arrow
        self.arrow_pos = np.array([50.0, self.height - 50.0], dtype=np.float32)
        self.arrow_vel = np.array([0.0, 0.0], dtype=np.float32)

        observation = self._get_obs()
        info = {} # distinct from obs, used for debugging
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Execute one action step. 
        For this env, one 'step' is ONE SHOT (simulation runs to completion).
        """
        angle = action[0]
        power = action[1]

        # Calculate Velocity Vector
        rad_angle = math.radians(angle)
        self.arrow_vel[0] = math.cos(rad_angle) * (power * 0.5)
        self.arrow_vel[1] = -math.sin(rad_angle) * (power * 0.5)

        # Simulate Flight
        terminated = False # Hit something
        truncated = False  # Time limit (not used here)
        hit_target = False
        
        # Physics Loop (Simulate entire flight instantly for the math)
        # Note: If rendering, we slow this down visually, but mathematically it happens in one step
        while True:
            self.arrow_pos += self.arrow_vel
            self.arrow_vel[1] += self.gravity

            # Render if human is watching
            if self.render_mode == "human":
                self._render_frame()

            # Check Collisions
            dist = math.hypot(self.arrow_pos[0] - self.target_pos[0], self.arrow_pos[1] - self.target_pos[1])
            
            # 1. Hit Target
            if dist < self.target_radius + 5:
                hit_target = True
                terminated = True
                break
            
            # 2. Hit Wall/Ground (Out of bounds)
            if self.arrow_pos[0] > self.width or self.arrow_pos[1] > self.height:
                terminated = True
                break

        # Calculate Reward
        if hit_target:
            reward = 100.0
        else:
            # Shaping: Negative reward based on distance to target
            reward = -1.0 * (dist / 10.0)

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # --- NORMALIZATION FIX ---
        # We divide by width/height so the AI sees coordinates as 0.0 to 1.0
        # instead of 0 to 800. This is crucial for learning!
        return np.array([
            self.arrow_pos[0] / self.width,
            self.arrow_pos[1] / self.height,
            self.target_pos[0] / self.width,
            self.target_pos[1] / self.height
        ], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Archery RL")
            self.clock = pygame.time.Clock()

        self.screen.fill((0, 0, 0))
        # Draw Target
        pygame.draw.circle(self.screen, (255, 0, 0), self.target_pos.astype(int), self.target_radius)
        # Draw Arrow (Simple dot for now to keep render fast)
        pygame.draw.circle(self.screen, (0, 255, 0), self.arrow_pos.astype(int), 5)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()