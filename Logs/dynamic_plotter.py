import matplotlib.pyplot as plt
import numpy as np
import re
import time

### Got dis from stackoverflow :D
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


normalize = True
smooth_val = 150

plt.ion()

while(True):
    log_dnn = open('ddqn_log.txt','r')

    rewards = []
    losses = []
    wins = []
    epsilons = []
    x = []

    for line in log_dnn:
        splits = re.split("[:\t\n+]",line)
        rewards.append(float(splits[2]))
        losses.append(float(splits[4]))
        wins.append(float(splits[6]))
        epsilons.append(float(splits[8]))
        x.append(int(splits[0]))
    
    log_dnn.close()

    # x = [i for i in range(len(reward))]
    reward = np.asarray(rewards)
    losses = np.asarray(losses)
    wins = np.asarray(wins)
    epsilons = np.asarray(epsilons)
    x = np.asarray(x)

    if normalize==True:
        rewards = NormalizeData(rewards)
        losses = NormalizeData(losses)
        wins = NormalizeData(wins)
        epsilons = NormalizeData(epsilons)
    
    l1, = plt.plot(x, smooth(reward,smooth_val), 'r-', lw=1.5,label="Reward")
    l2, = plt.plot(x, smooth(losses,smooth_val),'b-',lw=1.5,label="Loss")
    l3, = plt.plot(x, smooth(wins,smooth_val),'g-',lw=1.5,label="Win Rate")
    l4, = plt.plot(x, smooth(epsilons,smooth_val),'y-',lw = 1.5,label="Epsilon")
    plt.draw()
    plt.pause(1)
    plt.clf()
    