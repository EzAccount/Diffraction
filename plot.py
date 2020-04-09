import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

with open("data/") as f:
    for line in f:
        temp_x, temp_y = [float(num) for num in line.split()]
        x = np.append(x, temp_x)
        y = np.append(y, temp_y)
print(np.size(x), np.size(y))
plt.plot(x,y)

plt.show()
