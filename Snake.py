import numpy as np
import random
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import h5py
import os

# Constants
GRID_SIZE = 10  
BATCH_SIZE = 64
EPISODES = 50
MAX_TIMESTEPS = 1000

# Snake game class with added functionalities for saving/loading state and improvements
class SnakeGame:
    def __init__(self, width=GRID_SIZE, height=GRID_SIZE):
        self.width = width
        self.height = height
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()

    def _generate_food(self):
        #food pos
        while True:
            food_x = random.randint(0, self.width - 1)
            food_y = random.randint(0, self.height - 1)
            if (food_x, food_y) not in self.snake_position:
                return (food_x, food_y)

    def _get_state(self):
        """Return the current game state as a numpy array."""
        state = np.zeros((self.height, self.width))
        for x, y in self.snake_position:
            state[x, y] = 1  # Snake body part is marked as 1
        state[self.food_position] = 2  # Food is marked as 2
        return state

    def _calculate_distance(self):
        """Calculate Manhattan distance from snake head to food."""
        head_x, head_y = self.snake_position[0]
        food_x, food_y = self.food_position
        return abs(head_x - food_x) + abs(head_y - food_y)

    def step(self, action):
        """Take a step in the game based on the action and return state, reward, and done status."""
        self._update_direction(action)
        head_x, head_y = self.snake_position[0]
        new_head_x = (head_x + self.direction[0]) % self.width
        new_head_y = (head_y + self.direction[1]) % self.height
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Check for collisions
        if self._is_collision(new_head_x, new_head_y):
            reward = -10
            print(f"Collision! Reward: {reward}")  # Log the reward for collision
            return self._get_state(), reward, True, {"score": self.score, "reward": reward}

        # Calculate the current distance to food stuffs
        current_distance = self._calculate_distance()

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()
            self.score += 1
            reward = 10  # Reward for eating food
            print(f"Ate food! Reward: {reward}")  # Log the reward for eating food
        else:
            reward = self._calculate_movement_reward(current_distance)
            print(f"Moved. Reward: {reward}")  # Log the reward for movement
            self.snake_position.pop()  # Remove tail to maintain length

        self.previous_distance = current_distance
        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        """Update the snake's direction based on the chosen action."""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = directions[action]

    def _is_collision(self, head_x, head_y):
        """Check if the snake collides with itself or walls."""
        return (
                head_x < 0 or head_x >= self.width or
                head_y < 0 or head_y >= self.height or
                (head_x, head_y) in self.snake_position[1:]
        )

    def _calculate_movement_reward(self, current_distance):
        """Calculate reward based on the movement (getting closer/farther from food)."""
        if current_distance < self.previous_distance:
            return 1  # Reward for getting closer to food
        elif current_distance > self.previous_distance:
            return -1  # Punishment for moving away from food
        return 0  # Neutral movement

    def reset(self):
        """Reset the game to"""
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()
        return self._get_state()

# DQNAgent Class with improved structure and functionality
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network model."""
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_size, self.state_size)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on the epsilon-greedy strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Replay previous experiences and train the model."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        """Save the model to a file."""
        self.model.save(name)

    def load(self, name):
        """Load a previously saved model."""
        if os.path.exists(name):
            self.model.load_weights(name)
            print(f"Loaded model weights from {name}")
        else:
            print("No saved model found, starting from scratch.")

# Main loop to train the agent
if __name__ == "__main__":
    game = SnakeGame()
    agent = DQNAgent(state_size=GRID_SIZE, action_size=4)

    # Load pre-existing model if available
    agent.load("snake_weights.h5")

    scores = []
    for episode in range(EPISODES):
        state = game.reset().reshape(1, GRID_SIZE, GRID_SIZE)
        total_reward = 0
        for time in range(MAX_TIMESTEPS):
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)
            next_state = next_state.reshape(1, GRID_SIZE, GRID_SIZE)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {episode} finished after {time} timesteps with score: {info['score']}")
                break
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        scores.append(info["score"])

    # Save the model and scores
    agent.save("snake_weights.h5")
    with open("scores.json", "w") as f:
        json.dump(scores, f)

    print(f"Training completed. Final scores: {scores}")
