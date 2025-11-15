import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from collections import deque
import os # Import os module
from keras.models import load_model # Moved load_model import here
import pip # Import pip for package installation
from google.colab import files # Import files for downloading
from tensorflow.keras.utils import plot_model # Import plot_model for architecture visualization

def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        pip.main(["install", package])
        print(f"{package} installed successfully.")

# Parameters
episodes = int(input("How many episodes to run? "))

# Input validation for batch size
while True:
    try:
        batch_size_input = int(input("Batch size (positive integer)? "))
        if batch_size_input > 0:
            batch_size = batch_size_input
            break
        else:
            print("Batch size must be a positive integer.")
    except ValueError:
        print("Invalid input. Please enter an integer.")

# Reward shaping parameters (made tunable)
REWARD_FOOD = 10  # Reward for eating food
REWARD_CLOSER = 1.25  # Reward factor for moving 1 unit closer to food
REWARD_AWAY = -1  # Penalty factor for moving 1 unit away from food
REWARD_COLLISION = -5 # Penalty for collision (wall or self)
REWARD_TIME_PENALTY = -2 # Penalty for taking too long to eat food
REWARD_SURVIVAL = 0.2 # Small reward for surviving a step

# Step limit for eating food (made tunable)
STEPS_LIMIT_NO_FOOD = 3 * 10 * 10 # width * height * 3

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
            return self._get_state(), REWARD_COLLISION, True, {"score": self.score, "reward": REWARD_COLLISION} # Used REWARD_COLLISION

        # Check self collision
        if new_head in self.snake_position:
            return self._get_state(), REWARD_COLLISION, True, {"score": self.score, "reward": REWARD_COLLISION} # Used REWARD_COLLISION

        # Move inital snake head
        self.snake_position.insert(0, new_head)

        reward = REWARD_SURVIVAL # Small positive reward for surviving a step (Used REWARD_SURVIVAL)
        if new_head == self.food_position:
            self.score += 1
            reward = REWARD_FOOD  # Increased Large reward eating food (Used REWARD_FOOD)
            self.food_position = self._generate_food()
            self.steps_since_food = 0 # Reset step counter on eating food
            # NEW: reset distance baseline to the new food
            self.previous_distance = self._distance(self.snake_position[0], self.food_position)
        else:
            self.snake_position.pop()  # Remove tail
            # Reward shaping: closer to food
            dist = self._distance(new_head, self.food_position)
            distance_change = self.previous_distance - dist # Positive if closer, negative if farther

            if distance_change > 0: # Moved closer
                reward += distance_change * REWARD_CLOSER
            elif distance_change < 0: # Moved farther
                # distance_change is negative. REWARD_AWAY is also negative (e.g., -1).
                # To apply a penalty proportional to distance_change, we multiply
                # distance_change by the absolute value of REWARD_AWAY.
                # This ensures the reward added is negative (a penalty).
                reward += distance_change * abs(REWARD_AWAY)
            # If distance_change is 0, no reward/penalty from this shaping

            self.previous_distance = dist


        return self._get_state(), reward, False, {"score": self.score, "reward": reward}

    def _update_direction(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        new_dir = directions[action]
        # Don't allow reversing direction unless score is 0
        if self.score > 0 and (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) == (0, 0):
            pass # Do nothing if reversing is not allowed
        else:
            self.direction = new_dir

    def reset(self):
        self.snake_position = [(self.width//2, self.height//2)]
        self.direction = (0, 1)  # start moving right
        self.food_position = self._generate_food()
        self.score = 0
        self.previous_distance = self._distance(self.snake_position[0], self.food_position) # Reset distance to the new food position
        self.steps_since_food = 0 # Initialize step counter
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

        self.memory = deque(maxlen=50000) # was 5000
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Corrected minimum epsilon to a higher value
        self.epsilon_decay = 0.998  # Increased epsilon decay
        self.learning_rate = 0.0005 # or 0.0003 # Reduced learning rate

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.train_start = 1

    def _build_model(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'), # Added convolutional layer
            Flatten(),
            Dense(512, activation='relu'), # Increased dense layer neurons
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < max(self.train_start, batch_size):
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Predict Q-values for current states using the policy model
        q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next states using the policy model (for action selection in Double DQN)
        q_next_policy = self.model.predict(next_states, verbose=0)

        # Predict Q-values for next states using the target model (for Q-value estimation)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        # Initialize targets with current Q-values
        targets = np.copy(q_values)

        # Calculate Double DQN targets
        # Get the action with the highest Q-value from the policy model for the next state
        best_next_actions = np.argmax(q_next_policy, axis=1)

        # Use the target model to estimate the Q-value for that best action
        # Q_target = reward + gamma * Q_target_model(next_state, argmax(Q_policy_model(next_state)))
        td_targets = rewards + self.gamma * q_next_target[np.arange(batch_size), best_next_actions] * (1 - dones)

        # Update the target for the action taken
        targets[np.arange(batch_size), actions] = td_targets

        self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    # Save/load model in H5 format
    def save(self, name):
        self.model.save(name) # Removed save_format='h5'

    def load(self, name):
        # from keras.models import load_model # Moved load_model import to the top
        self.model = load_model(name, safe_mode=False) # Load with safe_mode=False
        self.target_model = load_model(name, safe_mode=False) # Load target model as well with safe_mode=False
        self.update_target_model()


if __name__ == "__main__":
    # Install necessary packages for plotting model architecture
    install_package('pydot')
    install_package('graphviz')

    plt.ion()
    game = SnakeGame()
    state_shape = (game.height, game.width, 3)
    agent = DDQNAgent(state_shape, action_size=4)
    scores = []

    watch = True  # Set True to see AI live play during training, False to speed up training

    # Define save path in the local Colab environment
    SAVE_PATH = "snake_weights.keras" # Changed file extension to .keras
    SCORES_PATH = "scores.json"

    try:
        agent.load(SAVE_PATH)
        print("Loaded saved weights")
        # Load scores if they exist
        if os.path.exists(SCORES_PATH):
            with open(SCORES_PATH, "r") as f:
                scores = json.load(f)
    except Exception as e: # Catch any exception during loading
        print(f"No saved weights found or error loading: {e}, starting training from scratch")


    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Score")
    ax.set_xlim(0, episodes)
    # Fix for ValueError: max() iterable argument is empty when scores is empty
    upper_y_limit = max(scores) + 1 if scores else 1
    ax.set_ylim(0, max(15, upper_y_limit))  # Adjust max score axis as needed
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Training Scores Over Episodes")
    ax.legend()
    plt.show(block=False)

    for episode in range(episodes):
        state = game.reset()
        done = False
        step = 0

        while not done: # Removed fixed step limit
            action = agent.act(state)
            next_state, reward, done, info = game.step(action)

            # Add condition to end episode if no food is eaten for too long (used STEPS_LIMIT_NO_FOOD)
            if game.steps_since_food > STEPS_LIMIT_NO_FOOD:
                done = True
                reward += REWARD_TIME_PENALTY # Add a penalty for not eating for too long (Used REWARD_TIME_PENALTY)
                info['reward'] = reward # Update reward in info dictionary
                info['score'] = game.score # Update score in info dictionary

            # right before remember(...)
            # reward = float(np.clip(reward, -1.0, 1.0))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            step += 1

            if watch:
                game.render()

            if step % 5 == 0:
                agent.replay(batch_size)

        scores.append(info['score'])

        # Decay epsilon at the end of the episode Please note that the epsilon
        # controls the chance of exploration
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Update target network periodically
        if episode % 5 == 0:
            agent.update_target_model()

        # Save model periodically during training
        if (episode + 1) % 10 == 0:
            agent.save(f"snake_weights_ep{episode+1}.keras") # Changed file extension to .keras


        # Updating the plot (live updates)
        line.set_data(range(len(scores)), scores)
        ax.set_xlim(0, max(episodes, len(scores)))
        # Fix for ValueError: max() iterable argument is empty when scores is empty
        upper_y_limit_update = max(scores) + 1 if scores else 1
        ax.set_ylim(0, max(15, upper_y_limit_update))
        fig.canvas.draw()
        fig.canvas.flush_events()

        print(f"Episode {episode+1}/{episodes} - Score: {info['score']} - Epsilon: {agent.epsilon:.3f}")

    plt.ioff()
    # Plotting scores after training
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Scores Over Episodes")
    plt.savefig("training_scores.png")
    plt.show()


    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=4) # Added indent for readability
    agent.save(SAVE_PATH)

    # Define the filename for the model architecture plot
    file_to_download_model_arch = 'model_architecture.png'

    # Plot and save the model architecture
    plot_model(agent.model, to_file=file_to_download_model_arch, show_shapes=True, show_layer_names=True)
    print("Model architecture saved to model_architecture.png")

    # Download the model architecture file
    if os.path.exists(file_to_download_model_arch):
        files.download(file_to_download_model_arch)
        print(f"Downloading {file_to_download_model_arch}...")
    else:
        print(f"{file_to_download_model_arch} not found. Please ensure the model architecture was generated successfully.")
