import time
from dataclasses import dataclass
import os
import numpy as np
import torch
import tyro
import supersuit as ss

import imageio.v2 as imageio

# Import your environment and model
from temporalg_3d import PettingZooWrapper 
from models import PPOLSTMCommAgent 

@dataclass
class TestArgs:
    # Path to the trained model checkpoint (Required)
    exp_name = "ippo_ms32_nenv8nb8_auxMaskTime_seed1"
    seed_name = "seed1"
    ckpt_name = "model_step_96000000"
    model_path: str = f"./checkpoints/train_from_scratch/{exp_name}/{seed_name}/{ckpt_name}.pt"
    # Environment settings (Must match training)
    env_id: str = "TemporalG-v1"
    seed: int = 1
    num_envs: int = 1  # Keep at 1 for testing/recording to be clean
    
    n_words: int = 4
    image_size: int = 48
    max_steps: int = 32
    
    # Test specific settings
    num_test_episodes: int = 50
    record_video: bool = True
    log_dir: str = f"logs/{exp_name}/{seed_name}/{ckpt_name}"
    cuda: bool = True
    
    # If the environment renders explicitly, set this to true
    # Otherwise, we assume env.render() returns an RGB array
    render_mode: str = "rgb_array" 

def extract_dict(obs_batch, device):
    """
    Extracts visual and location data. 
    """
    obs = torch.tensor(obs_batch['image']).to(device).float()
    locs = torch.tensor(obs_batch['location']).to(device).float()
    masks = torch.tensor(obs_batch['mask']).to(device).squeeze(-1)
    return obs, locs, masks

def main(args: TestArgs):
    # 1. Setup Device and Seeds
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    # 2. Initialize Environment
    # We use headless=False or pass render_mode depending on your specific Env implementation.
    # Assuming PettingZooWrapper accepts render_mode or we use wrappers.
    # If your wrapper doesn't support render_mode in init, you might need to adjust this.
    def make_env():
        return PettingZooWrapper(headless=True, image_size=args.image_size, max_steps=args.max_steps)
    
    env = make_env()
    
    # Vectorize: Even for 1 env, we keep the supersuit wrapper for consistency with input shapes
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, args.num_envs, num_cpus=1, base_class="gymnasium")
    
    # 3. Initialize Agent
    # We must match the dimensions used in training
    agent = PPOLSTMCommAgent(num_actions=3, 
                            n_words=args.n_words, 
                            embedding_size=64, 
                            num_channels=3, 
                            image_size=args.image_size).to(device)
    
    # 4. Load Checkpoint
    print(f"Loading model from {args.model_path}...")
    try:
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    agent.eval()
    
    os.makedirs(os.path.join(args.log_dir, "videos"), exist_ok=True)
    
    total_rewards = []
    lengths = []
    successes = [] # Assuming Reward > 0 implies success
    
    # Total agents in the vectorized environment
    total_agents = envs.num_envs

    for episode in range(args.num_test_episodes):
        print(f"Running Episode {episode + 1}/{args.num_test_episodes}...")
        
        # Reset Env
        obs_dict, _ = envs.reset(seed=args.seed + episode)
        obs, locs, masks = extract_dict(obs_dict, device)
        
        # Reset LSTM States
        lstm_state = (
            torch.zeros(agent.lstm.num_layers, total_agents, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, total_agents, agent.lstm.hidden_size).to(device),
        )
        
        # Reset Communication (Silence at t=0)
        r_messages = torch.zeros(total_agents, dtype=torch.int64).to(device)
        dones = torch.zeros(total_agents).to(device)
        
        episode_reward = np.zeros(total_agents)
        frames = []
        step_count = 0
        
# ... inside the episode loop ...
        
        while True:
            step_count += 1
            

            if args.record_video:
                try:
                    frame = envs.render() 
                    if frame is None:
                        frame = envs.unwrapped.envs[0].render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print("Cannot render")

            # --- 2. AGENT ACTION ---
            with torch.no_grad():
                action, _, _, s_message, _, _, _, _, _, next_lstm_state = agent.get_action_and_value(
                    (obs, locs, r_messages), 
                    lstm_state, 
                    dones
                )
            
            # --- 3. ENVIRONMENT STEP ---
            env_action = action.cpu().numpy()
            
            # Note: We capture 'next_obs_dict' here to use in the NEXT loop iteration
            next_obs_dict, reward, terminations, truncations, infos = envs.step(env_action)
            
            # Update obs_dict for the next video frame capture
            obs_dict = next_obs_dict 

            # ... rest of the loop (metrics, obs extraction, etc.) ...
            episode_reward += reward
            obs, locs, next_masks = extract_dict(next_obs_dict, device)
            lstm_state = next_lstm_state
            
            done_np = np.logical_or(terminations, truncations)
            dones = torch.Tensor(done_np).to(device)

            # Message Exchange Logic
            s_msgs_reshaped = s_message.view(-1, 2)
            swapped_msgs = torch.flip(s_msgs_reshaped, dims=[1]).flatten()
            r_messages = swapped_msgs * next_masks

            if np.any(done_np):
                break
        
        # --- Metrics Recording ---
        # Since VecEnv auto-resets, the reward we summed is correct for the episode.
        # We average reward across agents (or take sum depending on your metric).
        avg_ep_reward = np.mean(episode_reward)
        total_rewards.append(avg_ep_reward)
        lengths.append(step_count)
        
        # Define Success
        is_success = 1 if avg_ep_reward == 1 else 0
        successes.append(is_success)

        # save video
        if args.record_video and len(frames) > 0:
            video_name = os.path.join(
                args.log_dir,
                f"episode_{episode+1}_rew_{avg_ep_reward:.2f}.mp4"
            )

            fps = 4.0

            # imageio expects RGB uint8 frames (which you already have)
            imageio.mimsave(
                video_name,
                frames,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,              # optional (0–10)
                macro_block_size=None,  # prevents resizing warnings
            )

    envs.close()

    # 6. Save Statistics to TXT
    avg_reward = np.mean(total_rewards)
    success_rate = np.mean(successes) * 100
    avg_len = np.mean(lengths)
    
    stats_path = os.path.join(args.log_dir, "test_results.txt")
    with open(stats_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Episodes: {args.num_test_episodes}\n")
        f.write(f"Average Reward: {avg_reward:.4f}\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write(f"Average Length: {avg_len:.2f}\n")
    
    print("-" * 30)
    print(f"Testing Complete.")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Results saved to {stats_path}")
    if args.record_video:
        print(f"Videos saved to {args.log_dir}/")
    print("-" * 30)

if __name__ == "__main__":
    # You can run this via command line: 
    # python test.py --model-path checkpoints/temporal_3d/seed3/model_step_100000.pt
    cli_args = tyro.cli(TestArgs)
    main(cli_args)