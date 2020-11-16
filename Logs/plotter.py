import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import numpy as np
import re

# Normalizes the values to 0-1 range for ease of plotting all in one
normalize = True
# Smoothness factor of the graph
smooth_val = 250

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

log_dnn = open('ddqn_log.txt','r')

rewards = []
losses = []
wins = []
epsilons = []
x = []

sns.set(style="darkgrid")
plt.xticks(size=14)
plt.yticks(size=14)   
sns.despine(left=True,bottom=True)
plt.xlabel("Number of Batches")

for line in log_dnn:
    splits = re.split("[:\t\n+]",line)
    rewards.append(float(splits[2]))
    losses.append(float(splits[4]))
    wins.append(float(splits[6]))
    epsilons.append(float(splits[8]))
    x.append(int(splits[0]))

# x = [i for i in range(len(reward))]
reward = np.asarray(rewards)
losses = np.asarray(losses)
wins = np.asarray(wins)
epsilons = np.asarray(epsilons)
x = np.asarray(x)[:-smooth_val]

### Got this from stackoverflow :D
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if normalize==True:
    rewards = NormalizeData(rewards)
    losses = NormalizeData(losses)
    wins = NormalizeData(wins)
    epsilons = NormalizeData(epsilons)

### Comment the required lines in both plot and legend, and set Normalize = False to get true isolated graphs
l1, = plt.plot(x, smooth(reward,smooth_val)[:-smooth_val], CB91_Blue, antialiased=True,lw=1.75,label="Reward")
l2, = plt.plot(x, smooth(losses,smooth_val)[:-smooth_val], CB91_Purple, antialiased=True,lw=1.75,label="Loss")
l3, = plt.plot(x, smooth(wins,smooth_val)[:-smooth_val], CB91_Violet, antialiased=True,lw=1.75,label="Win Rate")
l4, = plt.plot(x, smooth(epsilons,smooth_val)[:-smooth_val],CB91_Amber, linestyle="dashed",antialiased=True,lw = 1.75,label="Epsilon")
plt.legend(handles=[
                    #l1,
                    l2,
                    l3,
                    l4
        ])
plt.show()