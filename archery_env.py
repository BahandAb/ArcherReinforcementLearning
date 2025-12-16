import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class ArcheryGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(ArcheryGymEnv, self).__init__()
        
        self.width = 800
        self.height = 600
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.accuracy_label = "N/A"

        self.gravity = 0.5
        self.target_radius = 20
        
        # --- NEW: Store the launch angle for visualization ---
        self.launch_angle_rad = 0.0 
        self.start_pos = np.array([50.0, self.height - 50.0]) # Fixed start point
        # -----------------------------------------------------

        # Action Space: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )

        # Observation Space: -Inf to Inf
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_pos = np.array([
            self.np_random.integers(300, self.width - 50),
            self.np_random.integers(50, self.height - 200)
        ], dtype=np.float32)
        
        # Reset Arrow to start position
        self.arrow_pos = self.start_pos.copy()
        self.arrow_vel = np.array([0.0, 0.0], dtype=np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # Action Scaling
        raw_angle = action[0]
        angle_deg = np.interp(raw_angle, [-1, 1], [0, 85])
        raw_power = action[1]
        power_val = np.interp(raw_power, [-1, 1], [15, 60]) # STRONG: 15 to 60 | WEAK: 10 to 30

        rad_angle = math.radians(angle_deg)
        
        # --- SAVE ANGLE FOR RENDERER ---
        self.launch_angle_rad = rad_angle
        # -------------------------------

        self.arrow_vel[0] = math.cos(rad_angle) * power_val
        self.arrow_vel[1] = -math.sin(rad_angle) * power_val

        terminated = False
        hit_target = False
        
        while True:
            self.arrow_pos += self.arrow_vel
            self.arrow_vel[1] += self.gravity 

            if self.render_mode == "human":
                self._render_frame()

            dist = math.hypot(self.arrow_pos[0] - self.target_pos[0], self.arrow_pos[1] - self.target_pos[1])

            if dist < self.target_radius + 10:
                hit_target = True
                terminated = True
                break
            
            if (self.arrow_pos[0] > self.width or 
                self.arrow_pos[0] < 0 or 
                self.arrow_pos[1] > self.height or 
                self.arrow_pos[1] < -100):
                terminated = True
                break

        if hit_target:
            reward = 100.0
        else:
            reward = -1.0 * (dist / 100.0)

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.array([
            self.arrow_pos[0] / self.width,
            self.arrow_pos[1] / self.height,
            self.target_pos[0] / self.width,
            self.target_pos[1] / self.height
        ], dtype=np.float32)

    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Archery AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 120, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise KeyboardInterrupt 

        self.screen.fill((30, 30, 30))
        
        # --- DRAW ARCHER (The visualization you asked for) ---
        # We draw a line from the start position pointing in the direction of the launch angle
        # Length of the "Bow" line
        bow_length = 50 
        
        # Calculate end point of the bow line
        end_x = self.start_pos[0] + math.cos(self.launch_angle_rad) * bow_length
        end_y = self.start_pos[1] - math.sin(self.launch_angle_rad) * bow_length # Minus because Y is flipped in Pygame

        # Draw the "Arm/Bow" (Cyan Line)
        pygame.draw.line(self.screen, (0, 255, 255), self.start_pos, (end_x, end_y), 4)
        
        # Draw a base pivot point (White dot)
        pygame.draw.circle(self.screen, (255, 255, 255), self.start_pos.astype(int), 5)
        # -----------------------------------------------------

        # Draw Accuracy Watermark
        if self.font:
            text_surf = self.font.render(self.accuracy_label, True, (255, 255, 255))
            text_surf.set_alpha(40) 
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(text_surf, text_rect)

        # Draw Target
        pygame.draw.circle(self.screen, (255, 50, 50), self.target_pos.astype(int), self.target_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), self.target_pos.astype(int), self.target_radius - 10)
        
        # Draw Arrow
        pygame.draw.circle(self.screen, (50, 255, 50), self.arrow_pos.astype(int), 5)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()