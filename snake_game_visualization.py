import numpy as np
import random
import warnings
from collections import deque
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Filter out the Axes3D warning (this does not affect functionality)
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

# Enable eager execution to avoid symbolic tensor issues
tf.config.run_functions_eagerly(True)

# Get user inputs for training parameters
episodes = int(input("Type a number of how many episodes the DDQN should run for: "))
batch_size = int(input("Type a number that is a power of 2 for the batch size: "))

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def _generate_food(self):
        # If possible, restrict food generation to avoid the edges
        if self.width > 2 and self.height > 2:
            x_range = range(1, self.width - 1)
            y_range = range(1, self.height - 1)
        else:
            x_range = range(0, self.width)
            y_range = range(0, self.height)
        while True:
            food_x = random.choice(list(x_range))
            food_y = random.choice(list(y_range))
            if (food_x, food_y) not in self.snake_position:
                return (food_x, food_y)

    def _get_state(self):
        # Create a more informative state representation with separate channels
        state = np.zeros((self.height, self.width, 3))

        # Channel 0: Snake head
        head_x, head_y = self.snake_position[0]
        state[head_x, head_y, 0] = 1

        # Channel 1: Snake body
        for x, y in self.snake_position[1:]:
            state[x, y, 1] = 1

        # Channel 2: Food
        food_x, food_y = self.food_position
        state[food_x, food_y, 2] = 1

        return state

    def _get_danger_state(self):
        # Create a state representation that includes danger information
        head_x, head_y = self.snake_position[0]

        # Check danger in each direction (up, down, left, right)
        danger = [False, False, False, False]

        # Up
        if head_y - 1 < 0 or (head_x, head_y - 1) in self.snake_position[1:]:
            danger[0] = True

        # Down
        if head_y + 1 >= self.height or (head_x, head_y + 1) in self.snake_position[1:]:
            danger[1] = True

        # Left
        if head_x - 1 < 0 or (head_x - 1, head_y) in self.snake_position[1:]:
            danger[2] = True

        # Right
        if head_x + 1 >= self.width or (head_x + 1, head_y) in self.snake_position[1:]:
            danger[3] = True

        # Direction of food relative to head
        food_x, food_y = self.food_position
        food_direction = [
            1 if food_y < head_y else 0,  # Food is up
            1 if food_y > head_y else 0,  # Food is down
            1 if food_x < head_x else 0,  # Food is left
            1 if food_x > head_x else 0,  # Food is right
        ]

        # Current direction
        current_direction = [0, 0, 0, 0]
        if self.direction == (0, -1):  # Up
            current_direction[0] = 1
        elif self.direction == (0, 1):  # Down
            current_direction[1] = 1
        elif self.direction == (-1, 0):  # Left
            current_direction[2] = 1
        elif self.direction == (1, 0):  # Right
            current_direction[3] = 1

        # Return the full state
        return np.array(danger + food_direction + current_direction)

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
            return self._get_state(), -100, True, {"score": self.score, "reward": -100}

        # Check for collisions with snake's body
        if (new_head_x, new_head_y) in self.snake_position[1:]:
            return self._get_state(), -100, True, {"score": self.score, "reward": -100}

        # Update snake's new head position
        self.snake_position.insert(0, (new_head_x, new_head_y))

        # Check if the snake is going in circles without eating food
        self.steps_without_food += 1
        if self.steps_without_food > 100 + 5 * len(self.snake_position):  # Scale with snake length
            return self._get_state(), -50, True, {"score": self.score, "reward": -50}

        # Check for food
        if (new_head_x, new_head_y) == self.food_position:
            self.food_position = self._generate_food()  # Generate new food
            self.score += 1
            self.steps_without_food = 0
            reward = 100 * (1 + (self.score / 10))  # Increasing reward for more food
        else:
            reward = self._calculate_movement_reward()
            self.snake_position.pop()  # Remove tail to maintain length

        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        # Define actions: 0: up, 1: down, 2: left, 3: right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_direction = directions[action]
        # Prevent immediate reversal of direction
        if (new_direction[0] + self.direction[0] != 0) or (new_direction[1] + self.direction[1] != 0):
            self.direction = new_direction

    def _calculate_movement_reward(self):
        current_distance = self._calculate_distance()
        distance_change = self.previous_distance - current_distance
        self.previous_distance = current_distance

        if distance_change > 0:
            return 5  # Significant reward for moving closer to food
        elif distance_change < 0:
            return -2  # Stronger penalty for moving away from food
        return -1  # Small penalty to discourage inaction

    def reset(self):
        # Start snake at a random position for more variety
        center_x, center_y = self.width // 2, self.height // 2

        # Start closer to center with a small random offset
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)

        # Ensure within bounds
        start_x = max(0, min(self.width - 1, center_x + offset_x))
        start_y = max(0, min(self.height - 1, center_y + offset_y))

        self.snake_position = [(start_x, start_y)]
        self.direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])  # Random initial direction
        self.food_position = self._generate_food()
        self.score = 0
        self.steps_without_food = 0
        self.previous_distance = self._calculate_distance()
        return self._get_state()

    def render(self):
        # Create a visualization with clear distinction between head, body, and food
        display = np.zeros((self.height, self.width, 3))

        # Snake body (green)
        for x, y in self.snake_position[1:]:
            display[x, y] = [0, 1, 0]

        # Snake head (blue)
        head_x, head_y = self.snake_position[0]
        display[head_x, head_y] = [0, 0, 1]

        # Food (red)
        food_x, food_y = self.food_position
        display[food_x, food_y] = [1, 0, 0]

        plt.clf()
        plt.imshow(display)
        plt.title(f"Score: {self.score}")
        plt.draw()
        plt.pause(0.05)  # Shorter pause for smoother animation

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_shape = state_size  # (10, 10, 3) for a 10x10 grid with 3 channels
        self.action_size = action_size  # 4 possible directions
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.gamma = 0.99  # Higher discount factor for better long-term planning
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.05  # Lower minimum exploration
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # Lower learning rate for stability
        self.tau = 0.01  # Soft update parameter

        # Build the primary Q-network and the target network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Frequency (in steps) to update the target network
        self.update_target_freq = 100  # Update more frequently
        self.step_counter = 0

        # Track performance for adaptive learning
        self.recent_scores = deque(maxlen=50)
        self.best_score = 0

    def _build_model(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        # Use MSE loss for compatibility across TensorFlow versions
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Soft update of the target network."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.array(state).reshape(1, *self.state_shape)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, *self.state_shape))
        targets = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state

            target = self.model.predict(np.array([state]), verbose=0)[0]

            if done:
                target[action] = reward
            else:
                # Double Q-learning: select action using main network
                next_action = np.argmax(
                    self.model.predict(np.array([next_state]), verbose=0)[0]
                )

                # Evaluate action using target network
                q_future = self.target_model.predict(np.array([next_state]), verbose=0)[0][next_action]

                target[action] = reward + self.gamma * q_future

            targets[i] = target

        # Train in batch for efficiency
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        # Increment counter and update target network if needed
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()

    def adaptive_learning(self, episode, score):
        """Adjust hyperparameters based on performance."""
        self.recent_scores.append(score)

        # Update best score
        if score > self.best_score:
            self.best_score = score

        # Calculate rolling average
        avg_score = np.mean(self.recent_scores) if self.recent_scores else 0

        # Adaptive epsilon decay
        if avg_score > 5:  # If performing well, reduce exploration faster
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)

        # If performance plateaus, occasionally increase exploration
        if len(self.recent_scores) >= 20:
            if np.std(list(self.recent_scores)[-20:]) < 0.5 and episode % 50 == 0:
                self.epsilon = min(0.5, self.epsilon * 1.5)  # Temporary boost to exploration

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        try:
            self.model.load_weights(name)
            self.update_target_model()
            return True
        except:
            return False

if __name__ == "__main__":
    # Setup figures for visualization
    plt.ion()  # Interactive mode on for rendering
    plt.figure(figsize=(10, 8))

    # Setup game and agent
    game = SnakeGame()
    state_shape = (10, 10, 3)  # Updated state shape with 3 channels
    agent = DDQNAgent(state_size=state_shape, action_size=4)

    # Training metrics
    scores = []
    avg_scores = []
    rolling_avg = deque(maxlen=100)

    # Try to load existing model
    loaded = agent.load("snake_weights.h5")
    if loaded:
        print("Loaded existing model.")
        # Start with less exploration if loaded model
        agent.epsilon = 0.3
    else:
        print("No existing model found. Starting fresh training.")

    # Training loop
    for episode in range(episodes):
        state = game.reset()
        total_reward = 0

        # Maximum steps per episode - increase with episodes to allow longer games
        max_steps = min(2000, 500 + (episode // 50) * 100)

        # Performance stats for this episode
        steps = 0
        episode_rewards = []

        for time in range(max_steps):
            # Select action and take step
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)

            # Tracking
            total_reward += reward
            episode_rewards.append(reward)
            steps += 1

            # Store experience and update state
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Visual feedback for EVERY step - now always renders
            game.render()

            # Status in title
            plt.suptitle(f"Episode: {episode+1}/{episodes} | Score: {info['score']} | Step: {time} | Epsilon: {agent.epsilon:.3f}")

            # Learning step
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break

        # End of episode tracking
        score = info["score"]
        scores.append(score)
        rolling_avg.append(score)
        avg_score = np.mean(list(rolling_avg))
        avg_scores.append(avg_score)

        # Adaptive parameter adjustments
        agent.adaptive_learning(episode, score)

        # Progress reporting
        print(f"Episode: {episode+1}/{episodes} | Score: {score} | Steps: {steps} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

        # Save best models
        if score >= agent.best_score and episode > 50:
            agent.save(f"snake_weights_best.h5")

        # Regular checkpoint saving
        if (episode + 1) % 50 == 0:
            agent.save(f"snake_weights_checkpoint_{episode+1}.h5")
            with open(f"scores_checkpoint_{episode+1}.json", "w") as f:
                json.dump({"scores": scores, "avg_scores": avg_scores}, f)

    # Final saving and visualization
    plt.ioff()
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(scores)
    plt.plot(avg_scores, 'r')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')

    plt.subplot(2, 1, 2)
    plt.hist(scores, bins=20)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # Save final results
    with open("scores.json", "w") as f:
        json.dump({"scores": scores, "avg_scores": avg_scores}, f)
    agent.save("snake_weights.h5")
