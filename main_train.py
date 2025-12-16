import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env # Needed for multicore
import os
import datetime
import webbrowser
from tensorboard import program
from archery_env import ArcheryGymEnv
import multiprocessing

# 1. Setup Directories
models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

if __name__ == '__main__':
    # 2. ASK USER FOR CORES
    # Multiprocessing requires this block to prevent infinite loops on Windows
    max_cores = multiprocessing.cpu_count()
    print(f"Your system has {max_cores} CPU cores.")
    
    while True:
        try:
            n_envs = int(input(f"How many environments (cores) to run in parallel? (1-{max_cores}): "))
            if 1 <= n_envs <= max_cores:
                break
            print("Invalid number.")
        except ValueError:
            print("Please enter a number.")

    # 3. Create Environments
    if n_envs == 1:
        # Single Env (Standard)
        env = ArcheryGymEnv(render_mode=None)
        # We can only run check_env on a SINGLE environment, not a vectorized one
        print("Checking environment compatibility...")
        check_env(env)
        print("Environment is valid!")
    else:
        # Vectorized Env (Multicore)
        # This creates 'n_envs' copies of ArcheryGymEnv running at once
        print(f"Creating {n_envs} parallel environments...")
        env = make_vec_env(lambda: ArcheryGymEnv(render_mode=None), n_envs=n_envs)

    # 4. Create Model
    print("Initializing PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

    # --- AUTO-LAUNCH TENSORBOARD ---
    try:
        print("Launching TensorBoard...")
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logs_dir])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not auto-launch TensorBoard: {e}")

    # 5. Train
    TIMESTEPS = 0 # 1500000
    print(f"Training started for {TIMESTEPS} steps on {n_envs} cores...")
    model.learn(total_timesteps=TIMESTEPS)
    print("Training finished!")

    # 6. Save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"archery_ppo_{timestamp}"
    save_path = f"{models_dir}/{model_name}"
    model.save(save_path)
    print(f"Model saved: {save_path}.zip")

    input("Press Enter to exit...")