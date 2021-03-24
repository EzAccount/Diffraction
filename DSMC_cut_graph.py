import constants as c
import numpy as np
import matplotlib.pyplot as plt
temp_x_arr = np.array([])
temp_y_arr = np.array([])
with open(c.density_path, encoding='utf-16') as f:  #
    for line in f:
        temp_x, temp_y = [float(x) for x in line.split()]
        temp_x_arr = np.append(temp_x_arr, temp_x)
        temp_y_arr = np.append(temp_y_arr, temp_y)
plt.plot(temp_x_arr, temp_y_arr)
plt.plot(np.arange(-c.wave_width, c.wave_width, 1E-6), np.zeros(np.size(np.arange(-c.wave_width, c.wave_width, 1E-6),)))
plt.show()
