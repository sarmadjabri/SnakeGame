Edit
Copy code
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import json
from collections import deque

episodes = int(input("Type a number of how many episodes the DQN should run for: "))
batch_size = int(input("Type a number that is a power of 2 for the batch size: "))

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)  # Initial direction: right
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()

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

        # Check for wall collision (no wrapping)
        if new_head_x < 0 or new_head_x >= self.width or new_head_y < 0 or new_head_y >= self.height:
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Update snake's new head position
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Calculate current distance before checking for collisions or food
        current_distance = self._calculate_distance()

        # Check for collisions with snake's body
        if (new_head_x, new_head_y) in self.snake_position[1:]:
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()  # Generate new food
            self.score += 1
            reward = 20  # Reward for eating food
        else:
            reward = self._calculate_movement_reward(current_distance)
            self.snake_position.pop()  # Remove tail to maintain length

        self.previous_distance = current_distance
        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        # Define possible directions
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_direction = directions[action]

        # Prevent reversing direction
        if (self.direction[0] + new_direction[0] != 0) or (self.direction[1] + new_direction[1] != 0):
            self.direction = new_direction

    def _calculate_movement_reward(self, current_distance):
        # Calculate how much closer we are to the food
        distance_change = self.previous_distance - current_distance

        # Positive reward if getting closer to the food
        if distance_change > 0:
            return distance_change  # Reward proportional to the distance change (positive)

        # Negative reward if moving away from the food
        elif distance_change < 0:
            return distance_change  # Penalty proportional to the distance increase (negative)

        # Small penalty if no change in distance
        return -0.1  # Slight penalty to discourage staying still

    def reset(self):
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()
        return self._get state()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Use deque for memory efficiency
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_size, self.state_size, 1)))  # Adjust input shape
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape(1, self.state_size, self.state_size, 1)  # Reshape for prediction
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
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
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)

if __name__ == "__main__":
    game = SnakeGame()
    agent = DQNAgent(state_size=10, action_size=4)
    scores = []

    # Try loading the pre-trained model
    try:
        agent.load("snake_weights.hdf5")
        print("Loaded existing model ")
    except:
        print("No existing model found, starting fresh training from the beginning.")

    for episode in range(episodes):
        state = game.reset().reshape(1, 10, 10, 1)  # Reshape with channels dimension
        for time in range(1000):
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)
            print("Episode: {} | Score: {}, Reward: {}".format(episode, info["score"], info["reward"]))
            next_state = next_state.reshape(1, 10, 10, 1)  # Reshape with channels dimension
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode {} finished after {} timesteps".format(episode, time))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Append score for each episode
        scores.append(info["score"])

    # Save the scores to a file
    with open("scores.json", "w") as f:
        json.dump(scores, f)

    # Save the trained model weights
    agent.save("snake_weights.hdf5")
