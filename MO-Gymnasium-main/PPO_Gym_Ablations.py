import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os

torch.manual_seed(0)
np.random.seed(0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

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

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=10):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        return action, action_logprob

    def update(self, memory):
        old_states = torch.stack(list(memory.states)).detach()
        old_actions = torch.stack(list(memory.actions)).detach()
        old_logprobs = torch.stack(list(memory.logprobs)).detach()
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values, rewards)
            
            loss = actor_loss.mean() + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

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

def train(load_existing=False, lr=0.0003, gamma=0.99, eps_clip=0.2):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ppo = PPO(state_dim, action_dim, lr=lr, gamma=gamma, eps_clip=eps_clip)
    checkpoint_path = './lunar_lander_ppo.pth'
    
    if load_existing and os.path.exists(checkpoint_path):
        print("Loading existing model...")
        ppo.policy.load_state_dict(torch.load(checkpoint_path))
        ppo.policy_old.load_state_dict(torch.load(checkpoint_path))
        return [], []
    
    memory = Memory()
    max_episodes = 500
    max_timesteps = 1000
    update_frequency = 2000
    log_interval = 20
    running_reward = 0
    avg_length = 0
    timestep = 0
    episode_rewards = []
    avg_lengths = []
    
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            action, logprob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            episode_reward += reward
            
            if timestep % update_frequency == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        episode_rewards.append(episode_reward)
        avg_length += 0.05 * (t+1 - avg_length)
        avg_lengths.append(avg_length)
        
        if i_episode % log_interval == 0:
            print(f'Episode {i_episode}\t Avg Length: {avg_length:.2f}\t Running Reward: {running_reward:.2f}')
        
        if running_reward > 200:
            print("Solved!")
            torch.save(ppo.policy.state_dict(), checkpoint_path)
            break
    
    torch.save(ppo.policy.state_dict(), checkpoint_path)
    env.close()
    return episode_rewards, avg_lengths

def plot_training_results(episode_rewards, avg_lengths, param_name=None, param_value=None):
    if param_name:
        prefix = f"{param_name}_{param_value}"
        title_suffix = f"({param_name}={param_value})"
    else:
        prefix = "training"
        title_suffix = ""
    
    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards)
    plt.title(f'Training Rewards {title_suffix}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'{prefix}_rewards.png')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(avg_lengths)
    plt.title(f'Average Episode Length {title_suffix}')
    plt.xlabel('Episode')
    plt.ylabel('Avg Length')
    plt.savefig(f'{prefix}_lengths.png')
    plt.close()

def run_ablation(param_name, values):
    for value in values:
        print(f"\nRunning ablation study for {param_name}={value}")
        rewards, lengths = train(load_existing=False, **{param_name: value})
        plot_training_results(rewards, lengths, param_name, value)

if __name__ == '__main__':
    # Original training/evaluation
    if os.path.exists('./lunar_lander_ppo.pth'):
        print("Found existing model. Evaluating...")
    else:
        print("Starting training...")
        rewards, lengths = train()
        plot_training_results(rewards, lengths)
    
    # Ablation studies
    print("\nStarting ablation studies...")
    
    # Learning Rate ablation
    run_ablation('lr', [0.001, 0.01, 0.0003])
    
    # Gamma ablation
    run_ablation('gamma', [0.5, 0.7, 0.999])
    
    # Clip Range ablation
    run_ablation('eps_clip', [0.1, 0.2, 0.3])
