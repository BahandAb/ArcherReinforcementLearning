import gymnasium as gym
from stable_baselines3 import PPO
import os
import webbrowser
from tensorboard import program
from archery_env import ArcheryGymEnv

models_dir = "models"
logs_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

try:
    print("Launching TensorBoard...")
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logs_dir])
    url = tb.launch()
    print(f"TensorBoard started at {url}")
    webbrowser.open(url)
except Exception as e:
    print(f"Could not auto-launch TensorBoard: {e}")

def get_saved_models(directory="models"):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []
    files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    files.sort(reverse=True) 
    return files

def main():
    models = get_saved_models()
    
    if not models:
        print("No models found. Run main_train.py first.")
        return

    print("\n--- Available Models ---")
    for i, model_file in enumerate(models):
        print(f"{i + 1}: {model_file}")
    
    while True:
        choice = input(f"\nSelect a model number (1-{len(models)}) or 'q' to quit: ")
        if choice.lower() == 'q': return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model_name = models[idx]
                break
            else: print("Invalid number.")
        except ValueError: print("Please enter a number.")

    model_path = os.path.join("models", selected_model_name)
    print(f"Loading {selected_model_name}...")
    
    env = ArcheryGymEnv(render_mode="human")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    
    # --- STATS TRACKING ---
    shots_fired = 0
    shots_hit = 0
    # ----------------------

    print("Running simulation... (Press Ctrl+C to stop)")

    try:
        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset on finish
            if terminated or truncated:
                # Update Stats
                shots_fired += 1
                if reward >= 100: # We used 100.0 for a hit in the env
                    shots_hit += 1
                
                # Calculate percentage
                accuracy = (shots_hit / shots_fired) * 100
                
                # Update the Environment's label so it draws on the next frame
                env.accuracy_label = f"{accuracy:.1f}%"
                
                print(f"Shot: {shots_fired} | Hit: {reward >= 100} | Accuracy: {accuracy:.3f}%")
                
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.close()


if __name__ == "__main__":
    main()