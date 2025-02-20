import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from collections import deque

# Get user inputs for training parameters
episodes = int(input("Type a number of how many episodes the DDQN should run for: "))
batch_size = int(input("Type a number that is a power of 2 for the batch size: "))

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def _generate_food(self):
        while True:
            food_x = random.randint(0, self.width - 1)
            food_y = random.randint(0, self.height - 1)
            if (food_x, food_y) not in self.snake_position:
                return (food_x, food_y)

    def _get_state(self):
        state = np.zeros((self.height, self.width))
        for x, y in self.snake_position:
            state[x, y] = 1  # Snake body
        state[self.food_position] = 2  # Food
        return state

    def _calculate_distance(self):
        head_x, head_y = self.snake_position[0]
        food_x, food_y = self.food_position
        return abs(head_x - food_x) + abs(head_y - food_y)

    def step(self, action):
        self._update_direction(action)
        head_x, head_y = self.snake_position[0]
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]

        # Check for wall collision
        if new_head_x < 0 or new_head_x >= self.width or new_head_y < 0 or new_head_y >= self.height:
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Update snake's new head position
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Check for collisions with snake's body
        if (new_head_x, new_head_y) in self.snake_position[1:]:
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()  # Generate new food
            self.score += 1
            reward = 10  # Reward for eating food
        else:
            reward = self._calculate_movement_reward()
            self.snake_position.pop()  # Remove tail to maintain length

        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        # Define actions: 0: up, 1: down, 2: left, 3: right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_direction = directions[action]
        # Prevent reversing direction immediately (sum of current and new must not be zero)
        if (new_direction[0] + self.direction[0] != 0) or (new_direction[1] + self.direction[1] != 0):
            self.direction = new_direction

    def _calculate_movement_reward(self):
        current_distance = self._calculate_distance()
        distance_change = self.previous_distance - current_distance
        self.previous_distance = current_distance

        if distance_change > 0:
            return 2    # Reward for moving closer to food
        elif distance_change < 0:
            return -0.5   # Penalty for moving away from food
        return -0.1     # Small penalty to discourage inaction

    def reset(self):
        # Start snake at a random position for more variety
        self.snake_position = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1))]
        self.direction = (0, 1)  # initial direction: right
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()
        return self._get_state()

    def render(self):
        plt.imshow(self._get_state(), cmap='hot', interpolation='nearest')
        plt.draw()
        plt.pause(0.1)

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # 10 for a 10x10 grid
        self.action_size = action_size  # 4 possible directions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.3  # Exploration probability
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001

        # Build the primary Q-network and the target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Frequency (in episodes) to update the target network
        self.update_target_freq = 5

    def _build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(10, 10, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(1, 10, 10, 1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Prepare state shapes
            state_input = state.reshape(1, 10, 10, 1)
            next_state_input = next_state.reshape(1, 10, 10, 1)

            # Use the main network to choose the best next action
            next_action = np.argmax(self.model.predict(next_state_input, verbose=0)[0])
            # Evaluate that action using the target network
            target_q_value = self.target_model.predict(next_state_input, verbose=0)[0][next_action]
            target = reward if done else reward + self.gamma * target_q_value

            # Predict current Q-values and update the Q-value for the taken action
            target_f = self.model.predict(state_input, verbose=0)
            target_f[0][action] = target

            # Train the main network
            self.model.fit(state_input, target_f, epochs=1, verbose=0)
            
        # Decay epsilon after replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()  # Ensure the target network is also updated

if __name__ == "__main__":
    plt.ion()  # Interactive mode on for rendering
    game = SnakeGame()
    agent = DDQNAgent(state_size=10, action_size=4)
    scores = []

    try:
        agent.load("snake_weights.h5")
        print("Loaded existing model.")
    except Exception as e:
        print("No existing model found. Starting fresh training.")

    for episode in range(episodes):
        state = game.reset().reshape(1, 10, 10, 1)
        for time in range(1000):
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)
            print(f"Episode {episode} | Time: {time} | Score: {info['score']} | Reward: {info['reward']}")
            next_state = next_state.reshape(1, 10, 10, 1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Render the game (can be commented out to speed up training)
            game.render()

            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(info["score"])

        # Update the target network every specified number of episodes
        if episode % agent.update_target_freq == 0:
            agent.update_target_model()
            print("Updated target network.")

    plt.ioff()
    plt.show()
    with open("scores.json", "w") as f:
        json.dump(scores, f)
    agent.save("snake_weights.h5")
