# Snake-Game: A Deep Reinforcement Learning Approach

This project implements a Deep Q-Network (DQN) agent to learn how to play the classic game of Snake. The agent learns through reinforcement learning, aiming to maximize its score by eating food while avoiding collisions.

## How to Experience the Project

There are two main ways to interact with this project:

### 1. Running the Double DQN Neural Network in Google Colab (Recommended)

This method allows you to train and visualize the DQN agent directly in your browser using Google Colab, a free cloud-based notebook environment. This is the best way to see the reinforcement learning in action.

**Steps:**

1.  **Open Google Colab:** Go to [https://colab.research.google.com/](https://colab.research.google.com/) and sign in with your Google account.
2.  **Create a New Notebook:** Click on `File > New notebook`.
3.  **Upload the Code:**
    *   Click on the folder icon on the left sidebar (`Files` section).
    *   Click the `Upload to session storage` icon (looks like a folder with an arrow pointing up).
    *   Upload the `snake_visualization.py` file from this repository.
    *   If you have a pre-trained model (e.g., `snake_weights.keras`), upload it to the same directory. The notebook is designed to load existing weights if found, otherwise, it will start training from scratch.
4.  **Copy and Paste the Code:** Copy the entire content of the `snake_visualization.py` file into the first code cell of your new Colab notebook.
5.  **Install Dependencies:** The provided code includes an `install_package` function to automatically install `pydot` and `graphviz` for visualizing the model architecture. Ensure this runs correctly.
6.  **Run the Notebook:** Execute all cells in the notebook. You can do this by clicking `Runtime > Run all`.
7.  **Interact with Prompts:** The notebook will prompt you for:
    *   **"How many episodes to run?"**: Enter the number of training episodes you want. More episodes mean more training time but potentially better performance.
    *   **"Batch size (positive integer)?"**: Enter the batch size for training the neural network (e.g., `16` or `32`).
8.  **Observe Training:** As the code runs, you will see:
    *   A live plot of the agent's scores over episodes.
    *   If `watch = True` is set in the code, a real-time visualization of the Snake game as the agent plays during training.
    *   Console output showing the episode number, score, and epsilon value (exploration rate).
9.  **Output Files:** After training, the notebook will generate:
    *   `snake_weights.keras`: The trained model weights (saved periodically and at the end).
    *   `scores.json`: A JSON file containing the scores from each training episode.
    *   `training_scores.png`: A plot visualizing the training scores.
    *   `model_architecture.png`: An image of the neural network's architecture.

### 2. Running the Python Files Locally (e.g., with Pygame Trinket)

For algorithms beyond the Deep Q-Network, you can use a local Python environment or online platforms like Pygame Trinket.

**Steps:**

1.  **Choose a File:** Select a Python file from this repository (e.g., `snakebot.py`).
2.  **Open Pygame Trinket:** Go to [https://trinket.io/features/pygame](https://trinket.io/features/pygame) or set up a local Python environment with Pygame installed.
3.  **Copy and Paste:** Copy the code from your chosen Python file and paste it into the Trinket editor or run it in your local environment.
4.  **Execute:** Run the code to see the algorithm play the Snake game.

## Important Notes

*   **Model Saving:** The Deep Q-Network model weights are saved in the `.keras` format (e.g., `snake_weights.keras`). This is a modern format for Keras models and is fully compatible with TensorFlow 2.x.
*   **Exploration vs. Exploitation (`self.epsilon`):** In the `DDQNAgent` class, the `self.epsilon` parameter controls the balance between exploration (trying new actions) and exploitation (using learned actions).
    *   Values closer to `1` mean more exploration.
    *   Values closer to `0` mean more exploitation.
    The `epsilon` value decays over time during training, allowing the agent to explore initially and then increasingly exploit its learned knowledge.

## Websites Used to Make This

*   [TensorFlow Tutorials: Reinforcement Learning (DQN)](https://www.tensorflow.org/tutorials/reinforcement_learning/dqn)
*   [Keras Documentation](https://keras.io)
*   [Python Official Documentation](https://docs.python.org)
*   [Denny Britz's Reinforcement Learning Blog](https://github.com/dennybritz/reinforcement-learning)
*   [Python Machine Learning GitHub](https://python-machinelearning.github.io)

Thank you for exploring this project! Contributions and feedback are welcome if you find this interesting.
