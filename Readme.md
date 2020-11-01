# Details:

## Game
Created the minesweeper game from scratch using python and numpy. Can take upto 7000 decisions per second when the renderer is off. the game component supports creation of any grid size with any number of bombs.

## Renderer
The backend of the game is connected with PyGame renderer. If this is turned on, the training time drastically increases and slows the process down. Color palette has been ripped off from colorhunt.

## Artificial Intelligence
#### Deep Q Learning

Deep Q learning takes in the state and tries to predict the action which will maximize the reward of the game. As opposed to traditional Q learning where all possible states are stored in memory, Deep Q tries to find a relation between input state and action, relieving the stress on memory requirements of the algorithm.<br/>
I've implemented two neural nets, one dense and another convolutional.
1. DNN: Takes in the flattened 10x10 matrix of the game and classifies it into 100 class action space with softmax
1. CNN: Takes in the matrix in 5x5 kernels and does the same classification as DNN

Both gave nearly the same results.<br/>
It took about 20 mins to train on 1200x4096 decisions.