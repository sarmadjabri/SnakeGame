import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from collections import deque

# Parameters
episodes = int(input("How many episodes to run? "))
batch_size = int(input("Batch size (power of 2)? "))

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def _generate_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake_position:
                return food

    def _get_state(self):
        # State as 3-channel input: snake body, snake head, food location
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for x, y in self.snake_position[1:]:
            state[x, y, 0] = 1.0  # body
        head_x, head_y = self.snake_position[0]
        state[head_x, head_y, 1] = 1.0  # head
        food_x, food_y = self.food_position
        state[food_x, food_y, 2] = 1.0  # food
        return state

    def _distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def step(self, action):
        self._update_direction(action)
        head_x, head_y = self.snake_position[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Check self collision
        if new_head in self.snake_position:
            return self._get_state(), -10, True, {"score": self.score, "reward": -10}

        # Move snake
        self.snake_position.insert(0, new_head)

        reward = 0
        if new_head == self.food_position:
            self.score += 1
            reward = 10  # Reward eating food
            self.food_position = self._generate_food()
        else:
            self.snake_position.pop()  # Remove tail
            # Reward shaping: closer to food
            dist = self._distance(new_head, self.food_position)
            if dist < self.previous_distance:
                reward = 1  # positive reward moving closer
            else:
                reward = -1  # penalty moving away
            self.previous_distance = dist

        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        new_dir = directions[action]
        # Don't allow reversing direction
        if (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) != (0, 0):
            self.direction = new_dir

    def reset(self):
        self.snake_position = [(self.width//2, self.height//2)]
        self.direction = (0, 1)  # start moving right
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._distance(self.snake_position[0], self.food_position)
        return self._get_state()

    def render(self):
        # Create a color-coded image:
        # body = 1, head = 3, food = 5 (for visual distinction)
        state = self._get_state()
        img = state[:,:,0]*1 + state[:,:,1]*3 + state[:,:,2]*5
        plt.imshow(img, cmap='viridis', interpolation='nearest')
        plt.title(f"Score: {self.score}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.05)


class DDQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size

        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.train_start = 1000

    def _build_model(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < max(self.train_start, batch_size):
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][a]

        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()


if __name__ == "__main__":
    plt.ion()
    game = SnakeGame()
    state_shape = (game.height, game.width, 3)
    agent = DDQNAgent(state_shape, action_size=4)
    scores = []

    watch = True  # Set True to see AI live play during training, False to speed up training

    try:
        agent.load("snake_weights.h5")
        print("Loaded saved model")
    except:
        print("No saved model found, starting training")

    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Score")
    ax.set_xlim(0, episodes)
    ax.set_ylim(0, 15)  # Adjust max score axis as needed
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Training Scores Over Episodes")
    ax.legend()
    plt.show(block=False)

    for episode in range(episodes):
        state = game.reset()
        done = False
        step = 0

        while not done and step < 500:
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step += 1

            if watch:
                game.render()

            if step % 4 == 0:
                agent.replay(batch_size)

        scores.append(info['score'])

        # Update target network periodically
        if episode % 5 == 0:
            agent.update_target_model()

        # Update plot live
        line.set_data(range(len(scores)), scores)
        ax.set_xlim(0, max(episodes, len(scores)))
        ax.set_ylim(0, max(15, max(scores) + 1))
        fig.canvas.draw()
        fig.canvas.flush_events()

        print(f"Episode {episode+1}/{episodes} - Score: {info['score']} - Epsilon: {agent.epsilon:.3f}")

    plt.ioff()
    plt.savefig("training_scores.png")
    plt.show()

    with open("scores.json", "w") as f:
        json.dump(scores, f)
    agent.save("snake_weights.h5")