import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os
import time # For unique filenames/folders

# Set seeds for reproducibility (optional, but good practice for comparisons)
# Note: Setting seeds globally might make different ablation runs too similar if
# the only difference is a hyperparameter. Consider setting seeds inside train
# or just before each train call if true independence between runs is needed.
# torch.manual_seed(0)
# np.random.seed(0)

def get_activation_fn(name: str):
    """Returns the activation function module based on its name."""
    if name.lower() == 'relu':
        return nn.ReLU
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid
    elif name.lower() == 'tanh':
        return nn.Tanh
    else:
        raise ValueError(f"Unknown activation function: {name}")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128], activation_fn=nn.ReLU):
        super(ActorCritic, self).__init__()

        # --- Actor Network ---
        actor_layers = []
        input_dim_actor = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(input_dim_actor, hidden_dim))
            actor_layers.append(activation_fn())
            input_dim_actor = hidden_dim
        # Actor head
        actor_layers.append(nn.Linear(input_dim_actor, action_dim))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)


        # --- Critic Network ---
        # Using separate layers for critic for potentially better stability
        critic_layers = []
        input_dim_critic = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(input_dim_critic, hidden_dim))
            critic_layers.append(activation_fn())
            input_dim_critic = hidden_dim
        # Critic head
        critic_layers.append(nn.Linear(input_dim_critic, 1))
        self.critic = nn.Sequential(*critic_layers)


    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        # Ensure action is returned as a Python int, logprob as a tensor
        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)
        # Ensure state_values is squeezed correctly (remove last dim of size 1)
        return action_logprobs, state_values.squeeze(-1), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=10,
                 hidden_dims=[128, 128], activation_fn=nn.ReLU, optimizer_name="Adam", entropy_coef=0.01, vf_coef=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef # Value function loss coefficient

        self.policy = ActorCritic(state_dim, action_dim, hidden_dims, activation_fn)
        self.optimizer = self._get_optimizer(optimizer_name, self.policy.parameters(), lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dims, activation_fn)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss() # Used for critic loss

    def _get_optimizer(self, name, params, lr):
        if name.lower() == "adam":
            return optim.Adam(params, lr=lr)
        elif name.lower() == "rmsprop":
            return optim.RMSprop(params, lr=lr)
        elif name.lower() == "sgd":
            return optim.SGD(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def select_action(self, state):
        with torch.no_grad():
            # Add batch dimension if state is a single observation
            if isinstance(state, np.ndarray):
                 state = torch.FloatTensor(state).unsqueeze(0)
            elif isinstance(state, torch.Tensor) and state.dim() == 1:
                 state = state.unsqueeze(0)

            action, action_logprob = self.policy_old.act(state)
        # action is sampled in act(), action_logprob corresponds to that action
        # Remove batch dimension from logprob if it exists
        return action, action_logprob.squeeze(0)

    def update(self, memory):
        # Convert lists in memory to tensors
        old_states = torch.stack(list(memory.states)).detach()
        old_actions = torch.stack(list(memory.actions)).detach()
        old_logprobs = torch.stack(list(memory.logprobs)).detach()

        # Monte Carlo estimate of returns (Advantages are calculated later)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards (often referred to as Returns here)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Lists to store losses from each epoch for averaging
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_entropy_losses = []
        epoch_total_losses = []

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values using the current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate Advantages = Returns - StateValues
            # Use state_values from the *current* policy but detached (as target)
            advantages = rewards - state_values.detach()
            # Note: Some implementations normalize advantages here:
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # Finding Surrogate Loss (Actor Loss component)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() # Mean over batch

            # Calculate Critic Loss (Value Function Loss)
            # Use the calculated returns as the target for the value function
            critic_loss = self.MseLoss(state_values, rewards) # MSELoss already includes mean

            # Calculate Entropy Bonus
            # We want to maximize entropy, so minimize negative entropy
            entropy_bonus = -self.entropy_coef * dist_entropy.mean() # Mean over batch

            # Combine losses for the total loss
            # loss = ActorLoss + CriticLossFactor * CriticLoss + EntropyBonusFactor * (-Entropy)
            loss = actor_loss + self.vf_coef * critic_loss + entropy_bonus # Total loss for this epoch

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping (helps prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # Store losses for this epoch (detach to avoid graph issues)
            epoch_actor_losses.append(actor_loss.item())
            epoch_critic_losses.append(critic_loss.item())
            # Store positive entropy value for easier interpretation in plots
            epoch_entropy_losses.append(dist_entropy.mean().item())
            epoch_total_losses.append(loss.item())


        # Copy new weights into old policy after optimization epochs
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Calculate average losses over the K epochs for logging/returning
        avg_total_loss = np.mean(epoch_total_losses)
        avg_actor_loss = np.mean(epoch_actor_losses)
        avg_critic_loss = np.mean(epoch_critic_losses)
        avg_entropy = np.mean(epoch_entropy_losses) # This is avg positive entropy

        return avg_total_loss, avg_actor_loss, avg_critic_loss, avg_entropy


class Memory:
    def __init__(self):
        self.actions = deque()
        self.states = deque()
        self.logprobs = deque()
        self.rewards = deque()
        self.is_terminals = deque()

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def __len__(self):
        # Return the number of transitions stored
        return len(self.states)


def train(env_name="LunarLander-v3", # Default to v2 as v3 might not be standard
          lr=0.0003,
          gamma=0.99,
          eps_clip=0.2,
          k_epochs=10,
          activation_fn_name="relu",
          hidden_dims=[128, 128],
          optimizer_name="Adam",
          update_frequency=2000, # Number of steps to collect before update
          max_episodes=500,
          max_timesteps=1000, # Max steps per episode
          log_interval=20,    # Print log every N episodes
          save_dir="ppo_results",
          run_name="default_run",
          random_seed=0):
    """Trains the PPO agent and collects metrics."""

    print(f"\n--- Starting Run: {run_name} ---")
    print(f"Parameters: lr={lr}, gamma={gamma}, eps_clip={eps_clip}, k_epochs={k_epochs}, "
          f"activation={activation_fn_name}, hidden_dims={hidden_dims}, optimizer={optimizer_name}, "
          f"update_freq={update_frequency}, seed={random_seed}")

    # Set seeds for this specific run for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create environment
    try:
        # Try creating the specified environment
        env = gym.make(env_name)
    except gym.error.NameNotFound:
        # Fallback to v2 if v3 (or other) isn't found
        print(f"Warning: Environment '{env_name}' not found. Trying 'LunarLander-v2'.")
        env_name = "LunarLander-v2"
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create PPO agent
    activation_fn = get_activation_fn(activation_fn_name)
    ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, k_epochs,
              hidden_dims, activation_fn, optimizer_name) # Pass necessary params

    # Setup logging and saving directories
    run_save_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_save_dir, 'ppo_model.pth')

    memory = Memory()
    running_reward = 0      # For tracking solving condition (EMA)
    avg_length_ema = 0      # Smoothed episode length (EMA)
    timestep = 0            # Global timestep counter across episodes
    solved_threshold = 200  # Reward threshold for considering the env solved
    solved = False          # Flag if solved

    # --- Logging lists ---
    episode_rewards = []    # Raw reward per episode
    episode_lengths = []    # Raw length (steps) per episode
    avg_lengths_ema_log = [] # Log of the smoothed length EMA

    # Loss tracking lists
    total_losses = []       # Avg total loss per update cycle
    actor_losses = []       # Avg actor loss per update cycle
    critic_losses = []      # Avg critic loss per update cycle
    entropy_log = []        # Avg policy entropy per update cycle
    update_indices = []     # Episode number at the time of each update


    # --- Training loop ---
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=random_seed + i_episode) # Seed env for each episode
        episode_reward = 0
        current_episode_length = 0 # Tracks length for the current episode

        # --- Episode loop ---
        for t in range(max_timesteps):
            timestep += 1
            current_episode_length += 1

            # Select action using the old policy (for interaction)
            action, logprob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # Check if episode ended

            # Store transition in memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            # Ensure states, actions, logprobs are stored as tensors
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)

            state = next_state
            episode_reward += reward

            # --- Policy Update Step ---
            # Update policy if enough steps have been collected
            if timestep % update_frequency == 0:
                if len(memory) > 0: # Ensure memory is not empty before update
                    # Perform PPO update using collected data
                    avg_total_loss, avg_actor_loss, avg_critic_loss, avg_entropy = ppo.update(memory)
                    # Clear memory after update
                    memory.clear_memory()

                    # Log losses and the episode index when the update occurred
                    total_losses.append(avg_total_loss)
                    actor_losses.append(avg_actor_loss)
                    critic_losses.append(avg_critic_loss)
                    entropy_log.append(avg_entropy)
                    update_indices.append(i_episode) # Log episode number at update time
                    # Reset timestep counter relative to update frequency? Optional.
                    # timestep = 0

            if done: # End episode if terminated or truncated
                break
        # --- End of Episode ---

        # --- Log episode metrics ---
        episode_rewards.append(episode_reward)
        episode_lengths.append(current_episode_length) # Store raw length

        # Update Exponential Moving Averages (EMA) for monitoring
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        avg_length_ema = 0.05 * current_episode_length + (1 - 0.05) * avg_length_ema
        avg_lengths_ema_log.append(avg_length_ema) # Store EMA of length for plotting

        # --- Print progress ---
        if i_episode % log_interval == 0:
             # Calculate average raw length over the logging interval for context
             avg_len_raw_interval = np.mean(episode_lengths[-log_interval:]) if len(episode_lengths) >= log_interval else np.mean(episode_lengths)
             print(f'Episode {i_episode}\t Avg Raw Len (last {log_interval}): {avg_len_raw_interval:.2f}\t'
                   f'Avg EMA Len: {avg_length_ema:.2f}\t Ep Reward: {episode_reward:.2f}\t Running Reward: {running_reward:.2f}')

        # --- Check for solving condition ---
        if running_reward > solved_threshold and not solved:
            print(f"########## Solved! Running reward ({running_reward:.2f}) exceeded {solved_threshold} ##########")
            torch.save(ppo.policy.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
            solved = True
            # Optionally break training early once solved
            # break

    # --- End of Training ---

    # Save final model if not solved during training
    if not solved:
        torch.save(ppo.policy.state_dict(), checkpoint_path)
        print(f"Finished training. Final model saved to {checkpoint_path}")

    env.close() # Close the environment

    # --- Plot and save results ---
    plot_training_results(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,        # Pass raw lengths
        avg_lengths_ema=avg_lengths_ema_log,    # Pass smoothed lengths log
        total_losses=total_losses,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropy_log=entropy_log,
        update_indices=update_indices,          # Pass indices for loss plotting
        save_dir=run_save_dir,                  # Directory to save plot
        run_name=run_name                       # Name for plot titles/filename
    )

    print(f"--- Finished Run: {run_name} ---")

    # --- Return performance metric for comparison ---
    # Use average reward of last N episodes as the metric
    metric_window = 50
    if len(episode_rewards) >= metric_window:
        final_metric = np.mean(episode_rewards[-metric_window:])
    elif episode_rewards: # Handle cases with fewer episodes than window
        final_metric = np.mean(episode_rewards)
    else:
        final_metric = -np.inf # Penalize runs that failed very early

    # Return raw episode rewards, raw lengths, and the final metric for ablation comparison
    return episode_rewards, episode_lengths, final_metric


def plot_training_results(episode_rewards,
                          episode_lengths,
                          avg_lengths_ema,
                          total_losses,
                          actor_losses,
                          critic_losses,
                          entropy_log,
                          update_indices,
                          save_dir,
                          run_name):
    """Plots rewards, episode lengths, and losses, then saves the figure."""

    print(f"Plotting results for run: {run_name}")
    if not episode_rewards: # Check if there's data to plot
        print(f"No episode reward data to plot for {run_name}. Skipping plot generation.")
        return

    num_episodes = len(episode_rewards)
    episodes = np.arange(1, num_episodes + 1)

    # Create figure with 3 rows, 2 columns of subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 14)) # Adjusted size
    fig.suptitle(f'Training Results: {run_name}', fontsize=18, y=0.99)

    # --- Row 1: Rewards and Smoothed Length ---
    # Ax[0, 0]: Episode Rewards
    axs[0, 0].plot(episodes, episode_rewards, label='Episode Reward', alpha=0.6, color='dodgerblue')
    # Add moving average line for rewards
    reward_window = 50 # Wider window for reward smoothing
    if num_episodes >= reward_window:
        moving_avg_reward = np.convolve(episode_rewards, np.ones(reward_window)/reward_window, mode='valid')
        axs[0, 0].plot(episodes[reward_window-1:], moving_avg_reward, label=f'{reward_window}-ep Moving Avg Reward', color='red', linewidth=2)
    axs[0, 0].set_title('Episode Rewards')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()

    # Ax[0, 1]: Smoothed Episode Length (EMA)
    if avg_lengths_ema:
        axs[0, 1].plot(episodes, avg_lengths_ema, label='Smoothed Length (EMA)', color='darkorange')
        axs[0, 1].set_title('Smoothed Episode Length (EMA)')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Avg Steps')
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        axs[0, 1].legend()
    else:
        axs[0, 1].set_title('Smoothed Length (No Data)')


    # --- Row 2: Raw Length and Total Loss ---
    # Ax[1, 0]: Raw Episode Length
    axs[1, 0].plot(episodes, episode_lengths, label='Episode Length', alpha=0.7, color='forestgreen')
    # Optional: Moving average for raw length
    len_window = 50
    if num_episodes >= len_window:
        moving_avg_len = np.convolve(episode_lengths, np.ones(len_window)/len_window, mode='valid')
        axs[1, 0].plot(episodes[len_window-1:], moving_avg_len, label=f'{len_window}-ep Moving Avg Length', color='darkgreen', linewidth=1.5)
    axs[1, 0].set_title('Raw Episode Length')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Number of Steps')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()

    # Ax[1, 1]: Total Loss (vs. Episode at Update)
    if total_losses and update_indices:
        axs[1, 1].plot(update_indices, total_losses, label='Total Loss', marker='.', linestyle='-', markersize=3, alpha=0.7, color='purple')
        # Moving average for loss
        loss_window = min(20, len(total_losses)) # Shorter window for loss
        if len(total_losses) >= loss_window:
             moving_avg_loss = np.convolve(total_losses, np.ones(loss_window)/loss_window, mode='valid')
             axs[1, 1].plot(update_indices[loss_window-1:], moving_avg_loss, label=f'{loss_window}-update Moving Avg Loss', color='indigo', linewidth=1.5)
        axs[1, 1].set_title('Total Loss per Update Cycle')
        axs[1, 1].set_xlabel('Episode Number at Update')
        axs[1, 1].set_ylabel('Avg Loss')
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)
        axs[1, 1].legend()
    else:
        axs[1, 1].set_title('Total Loss (No Data)')


    # --- Row 3: Loss Components (Actor, Critic, Entropy) ---
    # Ax[2, 0]: Actor Loss
    if actor_losses and update_indices:
        axs[2, 0].plot(update_indices, actor_losses, label='Actor Loss (Policy Gradient)', marker='.', linestyle='-', markersize=3, alpha=0.7, color='crimson')
        axs[2, 0].set_title('Actor Loss per Update Cycle')
        axs[2, 0].set_xlabel('Episode Number at Update')
        axs[2, 0].set_ylabel('Avg Loss')
        axs[2, 0].grid(True, linestyle='--', alpha=0.6)
        axs[2, 0].legend()
    else:
        axs[2, 0].set_title('Actor Loss (No Data)')

    # Ax[2, 1]: Critic Loss & Policy Entropy
    if critic_losses and update_indices:
        ax_critic = axs[2, 1]
        ax_entropy = ax_critic.twinx() # Create twin y-axis for entropy

        # Plot Critic Loss on primary axis
        line1, = ax_critic.plot(update_indices, critic_losses, label='Critic Loss (Value Function MSE)', marker='.', linestyle='-', markersize=3, alpha=0.7, color='darkgoldenrod')
        ax_critic.set_xlabel('Episode Number at Update')
        ax_critic.set_ylabel('Avg Critic Loss (MSE)', color='darkgoldenrod')
        ax_critic.tick_params(axis='y', labelcolor='darkgoldenrod')
        ax_critic.grid(True, linestyle='--', alpha=0.6)

        # Plot Entropy on secondary axis
        if entropy_log:
            line2, = ax_entropy.plot(update_indices, entropy_log, label='Policy Entropy', marker='.', linestyle='-', markersize=3, alpha=0.6, color='teal')
            ax_entropy.set_ylabel('Avg Policy Entropy', color='teal')
            ax_entropy.tick_params(axis='y', labelcolor='teal')
            # Combine legends from both axes
            lines = [line1, line2]
            ax_critic.legend(lines, [l.get_label() for l in lines], loc='best')
        else:
             ax_critic.legend(loc='best') # Legend for critic loss only if no entropy data

        ax_critic.set_title('Critic Loss & Policy Entropy per Update')

    else:
        axs[2, 1].set_title('Critic Loss & Entropy (No Data)')


    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.97]) # Adjust layout to prevent title overlap
    plot_filename = os.path.join(save_dir, f'training_plots_detailed.png')
    plt.savefig(plot_filename)
    print(f"Saved detailed plot to: {plot_filename}")
    plt.close(fig) # Close the figure to free memory


# ==============================================================================
# --- Main Execution Block for Ablation Studies ---
# ==============================================================================
if __name__ == '__main__':
    base_save_dir = "ppo_lunarlander_ablations_detailed" # Main folder for all results
    os.makedirs(base_save_dir, exist_ok=True)
    # Unique timestamp for this entire batch of ablation runs
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- Define Default Parameters (will be overridden in ablations) ---
    # These serve as the baseline for Phase 3 if Phase 1 doesn't yield a clear winner
    default_params = {
        "lr": 0.0003,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 10,
        "activation_fn_name": "relu",
        "hidden_dims": [128, 128],
        "optimizer_name": "Adam",
        "update_frequency": 2000,
        "max_episodes": 500, # Keep relatively short for faster ablations
        "random_seed": 42     # Use a fixed seed for the ablation set
    }
    print(f"Default parameters for reference: {default_params}")

    # ================== Phase 1: LR and Gamma Ablation ==================
    print("\n" + "="*20 + " Phase 1: LR and Gamma Ablation " + "="*20)
    lr_values = [0.0001, 0.0005, 0.001, 0.01] # Learning rates to test
    gamma_values = [0.99, 0.98, 0.97, 0.95, 0.90] # Discount factors to test
    results_lr_gamma = {} # Dictionary to store results: {(lr, gamma): metric}

    phase1_dir = os.path.join(base_save_dir, "1_lr_gamma")
    os.makedirs(phase1_dir, exist_ok=True)

    for lr_val in lr_values:
        for gamma_val in gamma_values:
            # Create a unique name for this specific run
            run_name = f"lr{lr_val}_gamma{gamma_val}_{run_timestamp}"
            current_params = default_params.copy() # Start with defaults
            current_params['lr'] = lr_val
            current_params['gamma'] = gamma_val

            # Run training with the current LR/Gamma combination
            _, _, final_metric = train(
                save_dir=phase1_dir, # Save results in the Phase 1 subfolder
                run_name=run_name,
                **current_params      # Pass all other params from defaults
            )
            # Store the performance metric for this combination
            results_lr_gamma[(lr_val, gamma_val)] = final_metric
            print(f"Result for lr={lr_val}, gamma={gamma_val}: Final Metric = {final_metric:.2f}")


    # ================== Phase 2: Find Best LR/Gamma ==================
    print("\n" + "="*20 + " Phase 2: Finding Best LR/Gamma " + "="*20)
    best_lr = default_params['lr']     # Initialize with defaults
    best_gamma = default_params['gamma'] # Initialize with defaults

    if not results_lr_gamma:
        print("Warning: No results from LR/Gamma ablation phase. Using default LR and Gamma.")
    else:
        # Filter out failed runs (metric = -inf) before finding max
        valid_results = {k: v for k, v in results_lr_gamma.items() if v > -np.inf}
        if not valid_results:
             print("Warning: All LR/Gamma runs failed or yielded -inf metric. Using default LR and Gamma.")
        else:
            # Find the (lr, gamma) tuple that produced the highest metric
            best_lr, best_gamma = max(valid_results, key=valid_results.get)
            best_metric = valid_results[(best_lr, best_gamma)]
            print(f"Best combination found: lr={best_lr}, gamma={best_gamma} "
                  f"with metric (Avg Reward Last 50) = {best_metric:.2f}")

    # Update base parameters for the next phase using the best found LR/Gamma
    base_params_for_phase3 = default_params.copy()
    base_params_for_phase3['lr'] = best_lr
    base_params_for_phase3['gamma'] = best_gamma

    print(f"\nUsing base parameters for Phase 3: {base_params_for_phase3}")


    # ================== Phase 3: Further Ablations ==================
    print("\n" + "="*20 + f" Phase 3: Further Ablations (using lr={best_lr}, gamma={best_gamma}) " + "="*20)

    # --- 3.1: Activation Functions ---
    print("\n--- Ablation 3.1: Activation Function ---")
    activation_names = ["relu", "sigmoid", "tanh"]
    phase3_act_dir = os.path.join(base_save_dir, "3_1_activation")
    for act_name in activation_names:
        # Avoid re-running the configuration already chosen as 'best'
        if act_name == base_params_for_phase3['activation_fn_name']:
             print(f"Skipping activation '{act_name}' as it's the baseline for Phase 3.")
             continue
        run_name = f"activation_{act_name}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
        train(
            activation_fn_name=act_name,
            save_dir=phase3_act_dir,
            run_name=run_name,
            **base_params_for_phase3 # Use best LR/Gamma, vary only activation
        )

    # --- 3.2: Network Architecture (Hidden Dimensions) ---
    print("\n--- Ablation 3.2: Network Architecture ---")
    hidden_dim_configs = {
        "small_1layer": [64],
        "medium_1layer": [128],
        "default_2layer": [128, 128], # Baseline likely
        "wide_2layer": [256, 128],
        "deep_3layer": [128, 128, 64]
    }
    phase3_arch_dir = os.path.join(base_save_dir, "3_2_architecture")
    for name, dims in hidden_dim_configs.items():
         if dims == base_params_for_phase3['hidden_dims']:
             print(f"Skipping architecture '{name}' ({dims}) as it's the baseline.")
             continue
         run_name = f"arch_{name}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
         train(
             hidden_dims=dims,
             save_dir=phase3_arch_dir,
             run_name=run_name,
             **base_params_for_phase3
         )

    # --- 3.3: PPO Clipping (eps_clip) ---
    print("\n--- Ablation 3.3: PPO Clipping Epsilon ---")
    eps_clip_values = [0.1, 0.2, 0.3]
    phase3_clip_dir = os.path.join(base_save_dir, "3_3_clipping")
    for eps_val in eps_clip_values:
        if eps_val == base_params_for_phase3['eps_clip']:
             print(f"Skipping eps_clip {eps_val} as it's the baseline.")
             continue
        run_name = f"epsclip_{eps_val}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
        train(
            eps_clip=eps_val,
            save_dir=phase3_clip_dir,
            run_name=run_name,
            **base_params_for_phase3
        )

    # --- 3.4: Update Frequency (Buffer Size before update) ---
    # Note: This is often called the 'rollout length' or 'buffer size' in PPO.
    print("\n--- Ablation 3.4: Update Frequency (Buffer Size) ---")
    # Practical values; very large buffers can be slow/memory intensive.
    update_freq_values = [500, 1000, 2000, 4000, 8000]
    phase3_freq_dir = os.path.join(base_save_dir, "3_4_update_frequency")
    print(f"Testing update frequencies (buffer sizes): {update_freq_values}")
    for freq_val in update_freq_values:
        if freq_val == base_params_for_phase3['update_frequency']:
             print(f"Skipping update_freq {freq_val} as it's the baseline.")
             continue
        run_name = f"updatefreq_{freq_val}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
        train(
            update_frequency=freq_val,
            save_dir=phase3_freq_dir,
            run_name=run_name,
            **base_params_for_phase3
        )

    # --- 3.5: Optimizer ---
    print("\n--- Ablation 3.5: Optimizer ---")
    optimizer_names = ["Adam", "RMSprop", "SGD"]
    phase3_opt_dir = os.path.join(base_save_dir, "3_5_optimizer")
    for opt_name in optimizer_names:
        if opt_name == base_params_for_phase3['optimizer_name']:
             print(f"Skipping optimizer '{opt_name}' as it's the baseline.")
             continue
        run_name = f"optimizer_{opt_name}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
        train(
            optimizer_name=opt_name,
            save_dir=phase3_opt_dir,
            run_name=run_name,
            **base_params_for_phase3
        )

    # --- 3.6: Explicit Buffer Size Values (if different from Update Freq tested) ---
    # This section runs specific buffer sizes requested, if they weren't covered in 3.4
    print("\n--- Ablation 3.6: Specific Buffer Sizes (Update Frequency) ---")
    buffer_size_values_requested = [1000, 5000, 10000, 50000] # Requested values (1k, 5k, 10k, 50k)
    phase3_buffer_dir = os.path.join(base_save_dir, "3_6_buffer_size")
    # Track frequencies already tested in section 3.4
    already_run_freqs = set(update_freq_values)
    # Also add the baseline frequency to the set of already run
    already_run_freqs.add(base_params_for_phase3['update_frequency'])

    for buf_val in buffer_size_values_requested:
        if buf_val in already_run_freqs:
             print(f"Skipping buffer size {buf_val} as it was already tested or is the baseline.")
             continue

        # If it's a new value, run it
        run_name = f"buffersize_{buf_val}_baseLR{best_lr}_baseGamma{best_gamma}_{run_timestamp}"
        train(
            update_frequency=buf_val, # Use buffer size as update frequency
            save_dir=phase3_buffer_dir,
            run_name=run_name,
            **base_params_for_phase3
        )
        already_run_freqs.add(buf_val) # Mark it as run


    print("\n" + "="*30 + " Ablation Studies Complete " + "="*30)
    print(f"All results saved in base directory: {base_save_dir}")
