import sys
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Define za envioronment
class SnakeGame:
    def __init__(self, width=80, height=80, food_reward=10, collision_penalty=-10):
        self.width = width
        self.height = height
        self.food_reward = food_reward
        self.collision_penalty = collision_penalty
        self.screen = np.zeros((height, width), dtype=np.uint8)
        self.snake = [(height // 2, width // 2)]
        self.direction = (0, 1)
        self.done = False
        self.food = None
    def reset(self):
        self.screen = np.zeros((self.height, self.width), dtype=np.uint8)
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = (0, 1)
        self.done = False
        self.spawn_food()
        return self.screen
    def step(self, action):
        if self.done:
            return self.screen, 0, self.done

        prev_head = self.snake[-1]
        new_head = (prev_head[0] + self.direction[0], prev_head[1] + self.direction[1])
        self.snake.append(new_head)
        if new_head in self.snake[:-1]:
            self.done = True
            reward = self.collision_penalty
        elif new_head[0] < 0 or new_head[0] >= self.height or new_head[1] < 0 or new_head[1] >= self.width:
            self.done = True
            reward = self.collision_penalty
        elif new_head == self.food:
            self.spawn_food()
            reward = self.food_reward
        else:
          -58,13 +59,13
        #def step(self, action):

    def get_action_direction(self, action):
        if action == 0:
            return (-1, 0)
        elif action == 1:
            return (1, 0)
        elif action == 2:
            return (0, -1)
        else:
            return (0, 1)

    def spawn_food(self):
        x, y = np.random.randint(0, self.height, 2)
        while (x, y) in self.snake:
            x, y = np.random.randint(0, self.height, 2)
        self.food = (x, y)
# Define the deep learning neural network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Agent:
    def __init__(self, state_dim, action_dim, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.memory = []
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters())
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            return torch.argmax(self.model(state)).item()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = np.random.choice(len(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in samples])
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states)
        next_q_values, _ = next_q_values.max(dim=1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
# Train loop
def train(env, agent, n_episodes=1000):
    scores = []
    for episode in range(n_episodes):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(state.flatten())
            next_state, reward, done = env.step(action)
            agent.remember(state.flatten(), action, reward, next_state.flatten(), done)
            agent.replay()
            state = next_state
            score += reward
        scores.append(score)
        if episode % 100 == 0:
            print(f"Episode {episode}: Score {score}")
    return scores
# Define the main
def main():
    # Set up the environment
    pygame.init()
    env = SnakeGame()
    # Set up the deep learning neural network
    state_dim = env.screen.shape[0] * env.screen.shape[1]
    action_dim = 4
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    target_update = 100
    agent = Agent(state_dim, action_dim, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, target_update)
    # Train the agent
    scores = train(env, agent, n_episodes=1000)
    # Plot the scores
    plt.plot(scores)
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
