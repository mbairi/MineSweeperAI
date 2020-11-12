# Details:

These are just to give you a rough overview of the project, more details about functions and variabls can be seen in the comments of my code.

## Game
Created the minesweeper game from scratch using python and numpy. Can take upto 7000 decisions per second when the renderer is off. the game component supports creation of any grid size with any number of bombs. <br/>
<b>Logic for the game is written in *game.py*</b><br/>

## Renderer
The backend of the game is connected with PyGame renderer. If this is turned on, the training time drastically increases and slows the process down. Color palette has been ripped off from colorhunt.<br/>
<b>Logic for the renderer is written in *renderer.py*</b><br/>

## Playable version

I have implemented a click and playable version incorporating both the above in object oriented fashion and included a clicker, and a solution display in the command prompt to debug.</br>
<b> Logic for the playable version is written in *playable.py*</b>
<br/>
<b> Just execute it in cmd to play it</b><br/>

    python playable.py

## Deep Q Learning

Deep Q learning takes in the state and tries to predict the action which will maximize the reward of the game. As opposed to traditional Q learning where all possible states are stored in memory, Deep Q tries to find a relation between input state and action, relieving the stress on memory requirements of the algorithm.<br/>

I have implemented Dueling DQN with help from RL Adventure repo<br/>

It takes roughly 20,000 x 2048 decisions to reach a rough winrate of 65 percent on a 6x6 board with 6 mines.<br/>

<b> Logic for training is written in *train_ddqn.py*</b> <br/>

    python train_ddqn.py

## Testing 

A simple python file for loading the pre trained module and testing its winrate and visualized test is written. <br/>
<b> Logic for testing is written in *tester.py*</br></b>

## Plotting

A module for reading the log file and printing a smoothened all in one normalized graph is present.
<br/>Note the plotter can run in <b>realtime while the module is being trained </b>as it reads from "./Logs/ddqn_log.txt" which is being written to in realtime by train_ddqn.py<br/>Re running of the plotter is required to update to latest epoch.<br/>
<b>Note: </b> the edge of the graph towards the last epoch always falls due to smoothening</br>
<b> Logic for the same is in *./Logs/plotter.py* </b><br/>

    cd Logs
    python plotter.py

