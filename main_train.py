import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from archery_env import ArcheryGymEnv

# 1. Instantiate the environment
# We use render_mode=None for training (super fast, no window)
env = ArcheryGymEnv(render_mode=None)

# 2. Check the environment
# This function checks if your environment follows the Gym API perfectly.
# If it crashes here, something is wrong with spaces or return types.
print("Checking environment compatibility...")
check_env(env)
print("Environment is valid!")

# 3. Create the Model
# MlpPolicy = Multi-Layer Perceptron (Standard Neural Net)
model = PPO("MlpPolicy", env, verbose=1)

# 4. Train the Model
print("Training started...")
model.learn(total_timesteps=200000) 
print("Training finished!")

# 5. Save the model
model.save("models/archery_ppo")

# --- VISUALIZATION ---
# Now we load the model and watch it play with graphics on
env = ArcheryGymEnv(render_mode="human")
obs, _ = env.reset()

print("Watching trained agent...")
for _ in range(10): # Watch 10 shots
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Shot Reward: {reward}")
    if terminated or truncated:
        obs, _ = env.reset()

env.close()