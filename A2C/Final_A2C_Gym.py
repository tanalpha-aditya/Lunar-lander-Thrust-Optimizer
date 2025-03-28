import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os
import wandb

torch.manual_seed(0)
np.random.seed(0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activation_function='relu', num_layers=2):
        super(ActorCritic, self).__init__()
        layers_actor = []
        layers_critic = []
        hidden_size = 128

        # Activation function selection
        if activation_function == 'relu':
            activation = nn.ReLU
        elif activation_function == 'sigmoid':
            activation = nn.Sigmoid
        else:
            raise ValueError(f"Activation function '{activation_function}' not supported.")

        # Actor layers
        layers_actor.append(nn.Linear(state_dim, hidden_size))
        layers_actor.append(activation())
        for _ in range(num_layers - 1): # Number of hidden layers
            layers_actor.append(nn.Linear(hidden_size, hidden_size))
            layers_actor.append(activation())
        layers_actor.append(nn.Linear(hidden_size, action_dim))
        layers_actor.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*layers_actor)

        # Critic layers
        layers_critic.append(nn.Linear(state_dim, hidden_size))
        layers_critic.append(activation())
        for _ in range(num_layers - 1): # Number of hidden layers
            layers_critic.append(nn.Linear(hidden_size, hidden_size))
            layers_critic.append(activation())
        layers_critic.append(nn.Linear(hidden_size, 1))
        self.critic = nn.Sequential(*layers_critic)

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values.squeeze(), dist_entropy

class A2C:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, optimizer_name='adam'):
        self.gamma = gamma
        self.policy = ActorCritic(state_dim, action_dim, activation_function=wandb.config.activation_function, num_layers=wandb.config.num_layers) # Access configs from wandb
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not supported.")
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, _ = self.policy.act(state)
        return action

    def update(self, memory):
        rewards = memory.rewards
        dones = memory.is_terminals

        # Compute returns
        returns = []
        discounted_return = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Convert to tensors
        old_states = torch.stack(list(memory.states))
        old_actions = torch.stack(list(memory.actions))

        # Evaluate current policy
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # Calculate advantages
        advantages = returns - state_values.detach()

        # Calculate losses
        actor_loss = -(logprobs * advantages).mean()
        critic_loss = self.MseLoss(state_values, returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), dist_entropy.mean().item()

class Memory:
    def __init__(self):
        self.actions = deque()
        self.states = deque()
        self.rewards = deque()
        self.is_terminals = deque()

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.is_terminals.clear()

def train(lr, gamma, run_name, activation_function, num_layers, optimizer_name, buffer_size): # Added new parameters
    env_id = "LunarLander-v3"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2C(state_dim, action_dim, lr=lr, gamma=gamma, optimizer_name=optimizer_name) # Pass optimizer_name
    checkpoint_path = f'./lunar_lander_a2c_lr_{lr}_gamma_{gamma}_act_{activation_function}_layers_{num_layers}_opt_{optimizer_name}_buffer_{buffer_size}.pth' # More descriptive path

    memory = Memory()
    max_episodes = 5000
    max_timesteps = buffer_size # Buffer size is now max_timesteps
    log_interval = 20
    running_reward = 0
    avg_length = 0
    episode_rewards = []

    # Initialize wandb run
    wandb.init(
        project="Lunar-Lander007", # Changed project name
        name=run_name,
        config={
            "learning_rate": lr,
            "gamma": gamma,
            "env_id": env_id,
            "seed": 0,
            "algorithm": "A2C",
            "activation_function": activation_function, # Log activation function
            "num_layers": num_layers, # Log number of layers
            "optimizer": optimizer_name, # Log optimizer
            "buffer_size": buffer_size # Log buffer size (max_timesteps)
        },
        api_key="3109e45ecb4ed9dad85e22af19852af76198d140"
    )
    config = wandb.config

    print(f"Training Run: {run_name}")
    print(f"LR: {lr}, Gamma: {gamma}, Activation: {activation_function}, Layers: {num_layers}, Optimizer: {optimizer_name}, Buffer: {buffer_size}")


    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        memory.clear_memory()

        for t in range(max_timesteps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.tensor(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # Update policy after each episode and get losses
        actor_loss, critic_loss, entropy_loss = agent.update(memory)

        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        episode_rewards.append(episode_reward)
        avg_length += 0.05 * (t+1 - avg_length)

        if i_episode % log_interval == 0:
            print(f'Episode {i_episode}\t Avg Length: {avg_length:.2f}\t Running Reward: {running_reward:.2f}')

        if running_reward > 200:
            print(f"Solved! at episode {i_episode} in Run: {run_name}")
            torch.save(agent.policy.state_dict(), checkpoint_path)
            break

        # Log metrics to wandb
        wandb.log({
            "episode_reward": episode_reward,
            "running_reward": running_reward,
            "avg_episode_length": avg_length,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "episode": i_episode
        })

    torch.save(agent.policy.state_dict(), checkpoint_path)
    env.close()
    wandb.finish()
    return episode_rewards

def evaluate(lr, gamma, activation_function, num_layers, optimizer_name, buffer_size, render=False): # Added params to evaluate
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, activation_function=activation_function, num_layers=num_layers) # Pass to ActorCritic
    checkpoint_path = f'./lunar_lander_a2c_lr_{lr}_gamma_{gamma}_act_{activation_function}_layers_{num_layers}_opt_{optimizer_name}_buffer_{buffer_size}.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    total_rewards = []
    num_episodes = 10

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action_probs = model.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode Reward: {episode_reward:.2f}")

    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return total_rewards

def plot_rewards(all_rewards, filename='ablation_study_rewards.png'):
    plt.figure(figsize=(12, 6))
    for label, rewards in all_rewards.items():
        plt.plot(rewards, label=label)
    plt.title('Ablation Study: Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    learning_rates = [0.0001, 0.0005, 0.001, 0.01]
    gammas = [0.99, 0.98, 0.97, 0.95, 0.90]
    activation_functions = ['relu'] # ['relu', 'sigmoid'] # Activation functions
    num_layers_options = [2] # [2, 3] # Number of layers
    optimizer_names =['adam'] #['adam', 'sgd'] # Optimizers
    buffer_sizes = [1000] # [1000, 5000] # Buffer sizes (max_timesteps)

    all_experiment_rewards = {}

    for lr in learning_rates:
        for gamma in gammas:
            for activation_function in activation_functions:
                for num_layers in num_layers_options:
                    for optimizer_name in optimizer_names:
                        for buffer_size in buffer_sizes:
                            run_name = f"A2C_lr={lr}_gamma={gamma}_act={activation_function}_layers={num_layers}_opt={optimizer_name}_buffer={buffer_size}"
                            print(f"\nStarting training for run: {run_name}")
                            rewards = train(lr, gamma, run_name, activation_function, num_layers, optimizer_name, buffer_size)
                            all_experiment_rewards[run_name] = rewards # Use run_name as label

    plot_rewards(all_experiment_rewards, filename='lunar_lander_ablation_rewards_extended.png')
    print("\nAblation study plots saved to lunar_lander_ablation_rewards_extended.png")

    # Example of evaluating the last run (you can modify to evaluate specific runs)
    print("\nEvaluating trained model with last hyperparameters...")
    evaluate(lr, gamma, activation_function, num_layers, optimizer_name, buffer_size)