import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
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
            state[x, y] = 1
        state[self.food_position] = 2
        return state

    def _calculate_distance(self):
        head_x, head_y = self.snake_position[0]
        food_x, food_y = self.food_position
        return abs(head_x - food_x) + abs(head_y - food_y)

    def step(self, action):
        self._update_direction(action)
        head_x, head_y = self.snake_position[0]
        new_head_x = (head_x + self.direction[0]) % self.width
        new_head_y = (head_y + self.direction[1]) % self.height
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Calculate current distance before checking for collisions or food
        current_distance = self._calculate_distance()

        # Check for collisions
        if self._is_collision(new_head_x, new_head_y):
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()
            self.score += 1
            reward = 10  # Reward for eating food
        else:
            reward = self._calculate_movement_reward(current_distance)
            self.snake_position.pop()  # Remove tail to maintain length

        self.previous_distance = current_distance
        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = directions[action]

    def _is_collision(self, head_x, head_y):
        return (
                head_x < 0 or head_x >= self.width or
                head_y < 0 or head_y >= self.height or
                (head_x, head_y) in self.snake_position[1:]
        )

    def _calculate_movement_reward(self, current_distance):
        if current_distance < self.previous_distance:
            return 1  # Reward for getting closer to food
        elif current_distance > self.previous_distance:
            return -1  # Punishment for moving away from food
        return 0  # Neutral movement

    def reset(self):
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._calculate_distance()
        return self._get_state()

    def render(self):
        plt.imshow(self._get_state(), cmap='hot', interpolation='nearest')
        plt.draw()
        plt.pause(0.1)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.state_size, self.state_size)))
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
    plt.ion()
    game = SnakeGame()
    agent = DQNAgent(state_size=10, action_size=4)
    batch_size = 64
    episodes = 1000  # Training for 1000 episodes
    scores = []

    for episode in range(episodes):
        state = game.reset().reshape(1, 10, 10)
        for time in range(1000):
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)
            print("Episode: {} | Score: {}, Reward: {}".format(episode, info["score"], info["reward"]))
            next_state = next_state.reshape(1, 10, 10)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            game.render()
            if done:
                print("Episode {} finished after {} timesteps".format(episode, time))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        scores.append(info["score"])
        plt.clf()
        plt.plot(scores)
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    # Save the scores to a file
    with open("scores.json", "w") as f:
        json.dump(scores, f)

    # Save the trained model weights
    agent.save("snake_weights.h5")
