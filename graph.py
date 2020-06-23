import matplotlib.pyplot as plt
import numpy as np
experiment_file = open('shadow_shock_exp.dat', 'r', encoding='UTF-16')
results_file = [open("data/result_84/intensity.dat","r"), open("data/result_65/intensity.dat","r")]
x = np.array([])
y = np.array([])
for f in experiment_file:
    temp_x, temp_y = [float(x) for x in f.split()]
    x = np.append(x, 0.001*temp_x)
    y = np.append(y, temp_y)
x_i = np.zeros((np.size(results_file), 600))
y_i = np.zeros((np.size(results_file), 600))
for j, file in enumerate(results_file):
    for i,line in enumerate(file):
        temp_x, temp_y = [float(x) for x in line.split()]
        x_i[j][i]= temp_x
        y_i[j][i]=temp_y

for j in range(np.size(results_file)):
    x_i[j] = x_i[j]
    y_i[j] = y_i[j]/np.max(y_i[j])
    plt.plot(x_i[j],y_i[j])

y = y/y[:1]
plt.plot(x,y, color='red')
plt.show()
