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
            nn.Softmax(dim=-1))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

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
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99):
        self.gamma = gamma
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
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

def train(load_existing=False):
    env_id = "LunarLander-v3"
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2C(state_dim, action_dim)
    checkpoint_path = './lunar_lander_a2c.pth'
    
    if load_existing and os.path.exists(checkpoint_path):
        print("Loading existing model...")
        agent.policy.load_state_dict(torch.load(checkpoint_path))
        return evaluate(20)
    
    memory = Memory()
    max_episodes = 5000
    max_timesteps = 1000
    log_interval = 20
    running_reward = 0
    avg_length = 0
    episode_rewards = []
    
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
        
        # Update policy after each episode
        agent.update(memory)
        memory.clear_memory()
        
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        episode_rewards.append(episode_reward)
        avg_length += 0.05 * (t+1 - avg_length)
        
        if i_episode % log_interval == 0:
            print(f'Episode {i_episode}\t Avg Length: {avg_length:.2f}\t Running Reward: {running_reward:.2f}')
        
        if running_reward > 200:
            print("Solved!")
            torch.save(agent.policy.state_dict(), checkpoint_path)
            break
    
    torch.save(agent.policy.state_dict(), checkpoint_path)
    env.close()
    return episode_rewards

def evaluate(num_episodes=10, render=True):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load('./lunar_lander_a2c.pth'))
    model.eval()
    total_rewards = []
    
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

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('lunar_lander_rewards_a2c.png')
    plt.close()

if __name__ == '__main__':
    if os.path.exists('./lunar_lander_a2c.pth'):
        print("Found existing model. Evaluating...")
        evaluate()
    else:
        print("Starting training...")
        rewards = train()
        plot_rewards(rewards)
        print("\nEvaluating trained model...")
        evaluate()