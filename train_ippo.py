import random
import time
from dataclasses import dataclass
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import supersuit as ss

# Import the environment from the updated file
from temporalg_3d import PettingZooWrapper 
from models import PPOLSTMCommAgent 

def extract_dict(obs_batch, device):
    """
    Extracts visual and location data. 
    Note: 'r_message' is NO LONGER in the env observation.
    We inject r_message manually in the training loop.
    """
    obs = torch.tensor(obs_batch['image']).to(device).float()
    locs = torch.tensor(obs_batch['location']).to(device).float()
    masks = torch.tensor(obs_batch['mask']).to(device).squeeze(-1)
    return obs, locs, masks

@dataclass
class Args:
    seed: int = 1
    env_id: str = "TemporalG-v1"
    total_timesteps: int = int(2e7) 
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    m_ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    log_every = 10
    
    n_words = 4
    image_size = 96

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    
    save_dir = f"checkpoints/temporal_3d/seed{seed}/"
    os.makedirs(save_dir, exist_ok=True)
    load_pretrained = False
    visualize_loss = True
    save_frequency = int(1e5)
    exp_name = f"temporal_3d_seed{seed}"
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "pickup_high_v1"
    wandb_entity: str = "maytusp"

if __name__ == "__main__":
    args = tyro.cli(Args)
    # Batch size logic needs to account for Agents. 
    # With Supersuit and vector envs, num_envs usually becomes (envs * agents).
    # Let's assume standard concatenated vec envs.
    
    run_name = args.exp_name
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name, monitor_gym=True, save_code=True)
    
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- ENV SETUP ---
    def make_env():
        return PettingZooWrapper(headless=True, image_size=args.image_size)

    # 1. Create Base Env
    env = make_env()
    
    # 2. Vectorize using Supersuit
    # concat_vec_envs_v1 results in a vector env of size (num_envs * num_agents)
    # Order is usually [Env0_Ag0, Env0_Ag1, Env1_Ag0, Env1_Ag1, ...]
    envs = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(envs, args.num_envs, num_cpus=12, base_class="gymnasium")

    # Recalculate batch sizes based on actual vectorized dimensions
    total_agents = envs.num_envs
    args.batch_size = int(total_agents * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # -----------------

    agent = PPOLSTMCommAgent(num_actions=3, 
                            n_words=args.n_words, 
                            embedding_size=64, 
                            num_channels=3, 
                            image_size=args.image_size).to(device)
                                    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((args.num_steps, total_agents, 3, args.image_size, args.image_size)).to(device)
    raw_locs = torch.zeros((args.num_steps, total_agents, 4)).to(device)
    
    # Store Received Messages (Input to Agent)
    r_messages = torch.zeros((args.num_steps, total_agents), dtype=torch.int64).to(device)
    
    # Store Sent Messages (Output from Agent)
    s_messages = torch.zeros((args.num_steps, total_agents), dtype=torch.int64).to(device)
    
    masks_store = torch.zeros((args.num_steps, total_agents), dtype=torch.int64).to(device)
    actions = torch.zeros((args.num_steps, total_agents)).to(device)
    action_logprobs = torch.zeros((args.num_steps, total_agents)).to(device)
    message_logprobs = torch.zeros((args.num_steps, total_agents)).to(device)
    rewards = torch.zeros((args.num_steps, total_agents)).to(device)
    dones = torch.zeros((args.num_steps, total_agents)).to(device)
    values = torch.zeros((args.num_steps, total_agents)).to(device)

    global_step = 0
    start_time = time.time()
    
    # --- RESET ---
    next_obs_dict, _ = envs.reset(seed=args.seed)
    next_obs, next_locs, next_masks = extract_dict(next_obs_dict, device)
    
    # Initialize "Last Received Message" buffer (t=0, silence)
    next_r_messages = torch.zeros(total_agents, dtype=torch.int64).to(device)
    
    next_done = torch.zeros(total_agents).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, total_agents, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, total_agents, agent.lstm.hidden_size).to(device),
    )

    running_ep_r = 0.0
    running_ep_l = 0.0
    running_num_ep = 0

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += total_agents
            obs[step] = next_obs
            raw_locs[step] = next_locs
            masks_store[step] = next_masks
            r_messages[step] = next_r_messages # Input for this step
            dones[step] = next_done

            # 1. Agent Forward Pass
            # We pass 'next_r_messages' which was calculated at the end of the PREVIOUS step
            with torch.no_grad():
                action, action_logprob, _, s_message, message_logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    (next_obs, next_locs, next_r_messages), 
                    next_lstm_state, 
                    next_done
                )
                values[step] = value.flatten()

            actions[step] = action
            s_messages[step] = s_message
            action_logprobs[step] = action_logprob
            message_logprobs[step] = message_logprob

            # 2. Environment Step (Physical Only)
            env_action = action.cpu().numpy()
            next_obs_dict, reward, terminations, truncations, infos = envs.step(env_action)
            
            # Extract next observation
            next_obs, next_locs, next_masks = extract_dict(next_obs_dict, device)
            
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done_np).to(device)

            # --- 3. MESSAGE EXCHANGE LOGIC (Crucial Part) ---
            # We must calculate what Agent A hears at t+1 based on what Agent B sent at t.
            # Batch layout: [Env0_A0, Env0_A1, Env1_A0, Env1_A1, ...]
            # We reshape to (Num_Envs, 2) to identify pairs.
            
            # Reshape Sent Messages
            s_msgs_reshaped = s_message.view(-1, 2)  # Shape: (Num_Envs, 2)
            
            # Swap: Agent 0 hears 1, Agent 1 hears 0
            # Flip along dim 1
            swapped_msgs = torch.flip(s_msgs_reshaped, dims=[1]).flatten() # Shape: (Total_Agents,)
            
            # Apply Mask:
            # next_masks is the mask for the current physical state (after stepping).
            # If mask is 0, communication is blocked.
            next_r_messages = swapped_msgs * next_masks
            
            # Note: next_r_messages is now ready for the START of the next loop iteration.
            # ------------------------------------------------

            if (global_step // total_agents) % args.save_frequency == 0:
                save_path = os.path.join(args.save_dir, f"model_step_{global_step}.pt")
                torch.save(agent.state_dict(), save_path)

            for info in infos:
                if "episode" in info:
                    running_ep_r += info["episode"]["r"]
                    running_ep_l += info["episode"]["l"]
                    running_num_ep += 1

            if args.visualize_loss and running_num_ep != 0 and (global_step // total_agents) % args.log_every == 0:
                writer.add_scalar("charts/episodic_return", running_ep_r / running_num_ep, global_step)
                writer.add_scalar("charts/episodic_length", running_ep_l / running_num_ep, global_step)
                running_ep_r = 0.0
                running_ep_l = 0.0
                running_num_ep = 0

        # Bootstrapping (Using next_r_messages for value estimation of terminal state)
        with torch.no_grad():
            next_value = agent.get_value(
                (next_obs, next_locs, next_r_messages),
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, 3, args.image_size, args.image_size))
        b_locs = raw_locs.reshape(-1, 4)
        b_r_messages = r_messages.reshape(-1)
        b_action_logprobs = action_logprobs.reshape(-1)
        b_s_messages = s_messages.reshape(-1)
        b_message_logprobs = message_logprobs.reshape(-1)
        b_actions = actions.reshape((-1))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        envsperbatch = total_agents // args.num_minibatches
        envinds = np.arange(total_agents)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, total_agents)

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, total_agents, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, new_action_logprob, action_entropy, _, new_message_logprob, message_entropy, newvalue, _ = agent.get_action_and_value(
                    (b_obs[mb_inds], b_locs[mb_inds], b_r_messages[mb_inds]),
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    b_s_messages.long()[mb_inds],
                )
                
                action_logratio = new_action_logprob - b_action_logprobs[mb_inds]
                action_ratio = action_logratio.exp()

                message_logratio = new_message_logprob - b_message_logprobs[mb_inds]
                message_ratio = message_logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * action_ratio
                pg_loss2 = -mb_advantages * torch.clamp(action_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Message loss
                mg_loss1 = -mb_advantages * message_ratio
                mg_loss2 = -mb_advantages * torch.clamp(message_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                mg_loss = torch.max(mg_loss1, mg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                action_entropy_loss = action_entropy.mean()
                message_entropy_loss = message_entropy.mean()
                loss = pg_loss + mg_loss - (args.ent_coef * action_entropy_loss) - (args.m_ent_coef * message_entropy_loss) + v_loss * args.vf_coef


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()


        if args.visualize_loss and (global_step // args.num_envs) % args.log_every == 0:
            SPS =  int(global_step / (time.time() - start_time))
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/action_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/message_loss", mg_loss.item(), global_step)
            writer.add_scalar("losses/action_entropy", action_entropy_loss.item(), global_step)
            writer.add_scalar("losses/message_entropy", message_entropy_loss.item(), global_step)
            writer.add_scalar("charts/SPS", SPS, global_step)
            print(f"SPS: {SPS}")

    final_save_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(agent.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    envs.close()
    writer.close()