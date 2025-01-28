# Snake-Game
In order to run the code to see for yourself just copy the code from the snake visualization file and run it because it will show a board and the DQN network playing the game as its using reinforcement learning

This will be laggy and CPU intensive as you are training the model.

If you want to see an algorithm which is more effective play this game then try using the snakebot.py file and run it in pygame trinket(https://trinket.io/features/pygame)
Copy paste the code from that file and run it inside and it should play the game fluently.

Recreation of google snake and a Deep Q Neural network to beat google snake.

The algorithm to beat snake game is a DQN learning neural network of which plays the game several times and saves its attempts.
The Deep Q neural network saves its attempts to disk(hopefully in due time).
The project is using tensorflow framework or keras because it is relatively easy to use.

[https://www.tensorflow.org/tutorials/reinforcement_learning/dqn](https://www.tensorflow.org/)
https://keras.io
https://docs.python.org
https://github.com/dennybritz/reinforcement-learning
https://python-machinelearning.github.io
these are the websites I used to make this

Anyway feel free to run it but a good thing to watch out for CPU usage. Thanks

Please note that AI reccomended that I do a few things, the following are listed:
  Making the reward function actually work after failed attempts
  Food function
  some graphing but it made mistakes

Please note that if you run snake.py remember that it saves all of its attempts to snake_weights.h5 file and h5 might have issues with the library itself.

In the code provided where it says self.epilison, It is a exploration for greed tradeoff from 0 to 1. 1 is the most explorative and 0 is the most greedy for points and wont learn or try new strategies.
