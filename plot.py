import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

with open("ExperimentHole.out") as f:
    for line in f:
        temp_x, temp_y = [float(x) for x in line.split()]
        x = np.append(x, temp_x)
        y = np.append(y, temp_y)
plt.plot(x,y)
plt.show()
