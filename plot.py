import numpy as np
import matplotlib.pyplot as plt

x = np.array([])
y = np.array([])

with open("data/NS_range1e-05step2.5e-07scaling1diversity0.2blurry_x.out") as f:
    for line in f:
        temp_x, temp_y = [float(num) for num in line.split()]
        x = np.append(x, temp_x)
        y = np.append(y, temp_y)
print(np.size(x), np.size(y))
plt.plot(x,y)

plt.show()
