# Snake-Game
Recreation of google snake and a Deep Q Neural network to beat google snake.
Source code is not fully mine as I did get help from numerous websites.

The algorithm to beat snake game is a DQN learning neural network of which plays the game several times and saves its attempts.
The Deep Q neural network saves its attempts to disk(hopefully in due time).
The project is using the keras library as it is relatively easy to use.

[https://www.tensorflow.org/tutorials/reinforcement_learning/dqn](https://www.tensorflow.org/)
https://keras.io
https://docs.python.org
https://github.com/dennybritz/reinforcement-learning
https://python-machinelearning.github.io


Anyway feel free to run it but a good thing to watch out for is the amount of CPU it uses Thanks!

Please note that AI reccomended that I do a few things, the following are listed:
  Making the reward function actually work after failed attempts
  Food function
  some graphing but it made mistakes

Please note that if you run googlesnake.py(its not google), remember that it saves all of its attempts to snake_weights.h5 file and h5 might have issues with the library itself.

In the code provided where it says self.epilison, the higher the number the more it will explore and the lower the number then the more it will chase points and so its a tradeoff with the epilison.
