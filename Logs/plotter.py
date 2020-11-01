import matplotlib.pyplot as plt
import numpy as np

log_cnn = open('cnn_log.txt','r')
log_dnn = open('dnn_log.txt','r')

epochs = 1200
cnn=[]
dnn=[]
x = []

for i in range(1200):
    cnn.append(float(log_cnn.readline()))
    dnn.append(float(log_dnn.readline()))
    x.append(i)

cnn = np.asarray(cnn)
dnn = np.asarray(dnn)
x = np.asarray(x)


### Got dis from stackoverflow :D
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

l1, = plt.plot(x, smooth(dnn,50), 'r-', lw=1.5)
l2, = plt.plot(x, smooth(cnn,50), 'g-', lw=1.5)
plt.legend(['DNN', 'CNN'])
plt.show()