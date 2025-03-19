# 🚀 Lunar Lander with Proximal Policy Optimization (PPO)

This project implements **Reinforcement Learning (RL)** using **Proximal Policy Optimization (PPO)** to train an agent to land a spacecraft safely in the **Lunar Lander** environment from OpenAI Gym.

---

## 📌 Project Overview
The objective is to develop an **optimal landing policy** that maximizes landing success while minimizing fuel consumption. PPO, a state-of-the-art **policy-gradient RL algorithm**, is used for stable and efficient learning.

---

## 🏗️ Implementation Details

### 🔹 Environment
- OpenAI Gym's **LunarLander-v2** environment.
- The agent must control thrust and orientation to land safely.
- Continuous state space and discrete action space.

### 🔹 RL Algorithm: PPO
- Implemented using **Stable Baselines3**.
- Uses a clipped surrogate objective to improve stability.
- Optimizes policy using an **actor-critic** approach.

### 🔹 Training Details
- **Library**: `Stable Baselines3`
- **Total Steps**: 1M+
- **Key Hyperparameters**: Learning rate, batch size, discount factor (γ)
- **Evaluation Metrics**:
  - **Average Reward** (higher is better)
  - **Landing Success Rate**
  - **Fuel Efficiency**

---

## 📈 Results & Research Questions
- How do **different reward structures** impact training performance?
- Can **hyperparameter tuning** improve PPO's stability?
- How does PPO compare with **DQN** and **A2C** for this task?

Expected outcome: A well-trained agent that lands successfully with optimized fuel consumption.

---

## 🛠️ Installation & Usage

### 🔧 Prerequisites
Ensure you have the following installed:
- Python 3.8+
- `gym`
- `stable-baselines3`
- `matplotlib` (for visualization)

### 📥 Installation
```bash
pip install gym[box2d] stable-baselines3 matplotlib
```

### 🚀 Training the PPO Agent
```python
from stable_baselines3 import PPO
import gym

# Create the environment
env = gym.make("LunarLander-v2")

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=1000000)

# Save the model
model.save("ppo_lunar_lander")
```

### 🎮 Running the Trained Agent
```python
model = PPO.load("ppo_lunar_lander")
obs = env.reset()

done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
```

---

## 📂 Project Structure
```
📂 lunar-lander-ppo
├── 📜 README.md  # Project documentation
├── 📜 train.py   # Script to train PPO agent
├── 📜 test.py    # Script to test trained agent
├── 📂 models     # Saved models
├── 📂 results    # Training logs and graphs
└── 📜 requirements.txt # Dependencies
```

---

## 📚 References
- OpenAI Gym: [https://github.com/openai/gym](https://github.com/openai/gym)
- Stable Baselines3: [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- PPO Paper: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

---

## 🎯 Future Work
- Compare PPO with **DQN** and **A2C**.
- Tune hyperparameters for better efficiency.
- Extend to **continuous action space** using **SAC or TD3**.

---

## 🤝 Contributing
Feel free to open issues or contribute to this repository!

🚀 Happy Reinforcement Learning!
