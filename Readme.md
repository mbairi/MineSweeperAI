# Introduction

This project was created from scratch using Python,Numpy to create the game logic, pygame to render the states and PyTorch to train the AI. It has given satisfactory results with the now dated DQN algorithm.<br/>

# Changelog

### - 02.20 by Xiao

- **Auto Flag added**

`flagged` state is now added, with value `-2` in the `state` array of `MineSweeper`.  

By setting `auto_flag=True` every time calling `choose()` function of `MineSweeper`, cells containing bomb with a 100% probability are automatically flagged and marked in purple. 

**Attention!** You may need to deal with the new possible value `-2` before feeding the `state` array to pretrained models. 

Below is an example on medium difficulty grid, after the very first random click: 

![auto_flag demo](./Example/auto_flag_demo.png)

- **Auto Play added**

Now the MineSweeper is able to play automatically by choosing one masked cell with a 0% probability of containing a bomb and *repeat until* no more cell satisfies this condition. 

Just set `auto_play=True` every time calling `choose()` function of `MineSweeper`.

Below is an example on medium difficulty grid, after the very first random click: 

![auto_play demo](./Example/auto_play_demo.png)

**Try it out and have fun!** Just run `playable.py` to discover the auto-play functionality.

Example of usage: `next_state, terminal, reward = env.choose(i, j, auto_flag=True, auto_play=True)`

Reminder: both `auto_flag` and `auto_play` take slightly more time to compute compare to the original `choose` function. 

### - 02.18 by Xiao

**More realistic rule sets are now added!** 

Instead of stupid initialization of bomb locations while reset, we've added two more possible rule set: 

- Windows XP rule set: the bomb is never located at the square of the first click (rules out 1 square)
- Windows 7 rule set: the bomb is never located at nor around the square of the first click (rules out 9 squares) 

In order to use new rule sets in your code, simply add one more parameter `rule` every time creating new instances of the `MinsSweeper()` class from `game.py`. 

Possible keywords are `default` (stupid initialization), `winxp` (Windows XP rule set) and `win7` (Windows 7 rule set). 

For example, `env = MineSweeper(width=10, height=10, bomb_no=9, rule='win7')`.

|difficulty|Windows XP rule|Windows 7 rule|
|---|---|---|
|easy (9*9 grid, 10 bombs)|91.63%|97.05%|
|medium (16*16 grid, 40 bombs)|78.04%|89.11%|
|hard (16*30 grid, 99 bombs)|40.07%|52.98%|

c.f. https://github.com/ztxz16/Mine

# Details:
These are just to give you a rough overview of the project, more details about functions and variables can be seen in the <b>comments of my code</b><br/>

## Requirements
    pytorch
    numpy
    matplotlib
    pygame

## Result:
<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/finalres.gif" width="500"/><br/>
white = zeros<br/>
dark gray = unexplored<br/>
<b> Note:</b> The game has been slowed down to do 2 decisions every 1 second<br/>

## Game
Created the minesweeper game from scratch using python and numpy. Can take upto 7000 decisions per second when the renderer is off. the game component supports creation of any grid size with any number of bombs. <br/>
<b>Logic for the game is written in *game.py*</b><br/>

## Renderer
The backend of the game is connected with PyGame renderer. If this is turned on, the training time drastically increases and slows the process down. Color palette has been ripped off from colorhunt.<br/>
<b>Logic for the renderer is written in *renderer.py*</b><br/>

## Playable version

I have implemented a click and playable version incorporating both the above in object oriented fashion and included a clicker, and a solution display in the command prompt to debug. Width, Height, and bomb numbers can be modified in the class variables</br>
<b> Logic for the playable version is written in *playable.py*</b>
<br/>
<b> Just execute it in cmd to play it</b><br/>

    python playable.py

<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/20grid.PNG" width ="400px"><br/>
Playable eversion of 20x20 grid with 20 mines<br/>

## Reinforcement Learning
### 1. Deep Q learning
Deep Q learning takes in the state and tries to predict the action which will maximize the reward of the game. As opposed to traditional Q learning where all possible states are stored in memory, Deep Q tries to find a relation between input state and action, relieving the stress on memory requirements of the algorithm.<br/>

I have implemented Dueling DQN with help from RL Adventure repo<br/>

It takes roughly 20,000 x 4096 decisions to reach a rough winrate of 60 percent on a 6x6 board with 6 mines.<br/>
Training on an i5 6200u took about 18 hours roughly.<br/>

<b> Logic for training is written in *train_ddqn.py*</b> <br/>

    python train_ddqn.py

### 2. Action Masking
Implemented action masking in softmax layer of the model to prevent it from choosing invalid actions, such as tiles which are already selected.<br/>
This modification helped the model learn faster as opposed to giving it a -ve reward on choosing invalid actions<br/>
### 3. Gradient Clipping ( Deprecated )
Added a clip to the gradients as I was facing vanishing and exploding gradients leading to nan values after few epochs of training <br/>
### 4. Soft Target Network update 
Due to this, gradient clipping is not required. The soft clipping solved the loss jumping to very high values.<br/>
### 5. Reward Based Epsilon Decay (RBED)
This implementation carries out epsilon decay based on the reward of the model. It decreases the epsilon value to a factor of itself (0.9x here) if a reward threshold is met, and then increases the reward threshold by a certain amount. The process is repeated and a minimum cap of 0.01 is set for epsilon.

## Testing 

A simple python file for loading the pre trained module and testing its winrate and visualized test is written. <br/>
<b> Logic for testing is written in *tester.py*</br></b>

    python tester.py

<b>win_tester: </b> function checks the number of wins in 1000 games <br/>
<b>slow_tester: </b> visualizes a single game of the game with the AI while printing rewards </br>

## Plotting

A module for reading the log file and printing a smoothened all in one normalized graph is present.
<br/>Note the plotter can run in <b>realtime while the module is being trained </b>as it reads from "./Logs/ddqn_log.txt" which is being written to in realtime by train_ddqn.py<br/>

Static Plotter:

    cd Logs
    python plotter.py

Dynamic Plotter ( Refreshes every second ) :

    cd Logs
    python dynamic_plotter.py

<b> Note:</b> <br/>
1. for both let 150 steps of the training run, cause smoothener needs a minimum of 150 steps.
1. The edge of the graph always falls off as a straight line due to smoothening factor. Don't panic ~_~
1. Everything is normalized from 0-1 just for my ease to see at a glance if the model is learning or not.


<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/AllinOneNormalized.png" width = "700px"/><br/>
(Above) Normalized graph to give rough overview of the increase and decrease of parameters.<br/>
(Below) Isolated graphs of the parameters with respect to their original scales.<br/>
<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/Loss.png" width ="700px"><br/>
<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/reward.png" width ="700px"><br/>
<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/winrate.png" width ="700px"><br/>
<img src="https://github.com/Manjunatha-b/MineSweeperAI/blob/master/Example/Epsilon.png" width ="700px"><br/>

