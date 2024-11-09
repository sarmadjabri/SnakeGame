import numpy as np
import random
import json
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import h5py
import heapq

# Constants for the game grid
WIDTH = 9
HEIGHT = 9
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT

class SnakeGame:
    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.snake_position = [(0, 0)]
        self.direction = (0, 1)  # Start moving to the right
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
            state[x, y] = 1  # Mark the snake body
        state[self.food_position] = 2  # Mark the food
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

        # Check for collisions
        if self._is_collision(new_head_x, new_head_y):
            reward = -10
            print(f"Collision! Reward: {reward}")  # Log the reward for collision
            return self._get_state(), reward, True, {"score": self.score, "reward": reward}

        # Calculate the current distance to food
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
        self.direction = DIRECTIONS[action]

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

class Pathfinding:
    def __init__(self):
        pass

    def a_star_search(self, start, goal, snake_body):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for direction in DIRECTIONS:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Valid move check (no body or wall)
                if (0 <= neighbor[0] < WIDTH and
                        0 <= neighbor[1] < HEIGHT and
                        neighbor not in snake_body):  # Valid move
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                        if neighbor not in [i[1] for i in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount rate
        self.epsilon = 0.5  # Exploration rate
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


def main():
    game = SnakeGame()
    agent = DQNAgent(state_size=9, action_size=4)  # Update state_size to 9 since the grid is 9x9
    pathfinder = Pathfinding()

    batch_size = 2
    episodes = 1000
    scores = []

    for episode in range(episodes):
        state = game.reset().reshape(1, 9, 9)  # Reshape to (1, 9, 9) since the grid is 9x9
        for time in range(1000):
            # Try A* pathfinding to get to the food
            start = game.snake_position[0]
            goal = game.food_position
            path = pathfinder.a_star_search(start, goal, game.snake_position)

            if path:
                # Move in the direction of the path found by A*
                next_move = path[1]
                action = DIRECTIONS.index((next_move[0] - start[0], next_move[1] - start[1]))
            else:
                # If A* can't find a path, fall back to DQN agent
                action = agent.act(state)

            next_state, reward, done, info = game.step(action)
            next_state = next_state.reshape(1, 9, 9)  # Reshape to (1, 9, 9)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("Episode {} finished after {} timesteps".format(episode, time))
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(info["score"])

    # Save the agent's scores and model
    with open("scores.json", "w") as f:
        json.dump(scores, f)

    agent.save("snake_weights.h5")

if __name__ == "__main__":
    main()
