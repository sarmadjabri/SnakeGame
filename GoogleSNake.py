import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0

    def _generate_food(self):
        while True:
            food_x = random.randint(0, self.width - 1)
            food_y = random.randint(0, self.height - 1)
            if (food_x, food_y) not in self.snake_position:
                return (food_x, food_y)

    def _get_state(self):
        state = np.zeros((self.height, self.width))
        for x, y in self.snake_position:
            state[x % self.height, y % self.width] = 1
        state[self.food_position[0] % self.height, self.food_position[1] % self.width] = 2
        return state

    def step(self, action):
        if action == 0:  # Up
            self.direction = (0, -1)
        elif action == 1:  # Down
            self.direction = (0, 1)
        elif action == 2:  # Left
            self.direction = (-1, 0)
        elif action == 3:  # Right
            self.direction = (1, 0)

        # Move snake
        head_x, head_y = self.snake_position[0]
        new_head_x, new_head_y = (head_x + self.direction[0]) % self.width, (head_y + self.direction[1]) % self.height
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Check for collision with wall or self
        if (new_head_x < 0 or new_head_x >= self.width or
                new_head_y < 0 or new_head_y >= self.height or
                (new_head_x, new_head_y) in self.snake_position[1:]):
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}
        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()
            self.score += 1
            return self._get_state(), 10, False, {"score": self.score, "reward": 10}
        else:
            # Reward for going to food
            food_distance = abs(new_head_x - self.food_position[0]) + abs(new_head_y - self.food_position[1])
            previous_distance = abs(self.snake_position[0][0] - self.food_position[0]) + abs(self.snake_position[0][1] - self.food_position[1])
            if food_distance < previous_distance:
                reward = 0.1
            else:
                reward = -0.1
            self.snake_position.pop()  # Remove the tail
            return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def reset(self):
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)
        self.food_position = self._generate_food()
        self.score = 0
        return self._get_state()

    def render(self):
        state = self._get_state()
        print(state)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 0.5  # Lower epsilon for more exploitation # Higher epsilon for more exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
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
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
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
    batch_size = 512  # Increased batch size
    episodes = 10000 #More run time for more attempts

    for episode in range(episodes):
        state = game.reset()
        state = state.reshape(1, 10, 10)
        for time in range(1000):
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)
            print("Score: {}, Reward: {}".format(info["score"], info["reward"]))
            next_state = next_state.reshape(1, 10, 10)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode {} finished after {} timesteps".format(episode, time))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    agent.save("snake_weights.h5")
