from tqdm import tqdm
import matplotlib.pyplot as plt
from n_functions import *


def Solve(file):
    print('Solving shockwave passthroughs')

    for j in tqdm(range(0, np.size(c.X), 1)):
        x = np.zeros(c.m + 1)
        tgA = np.zeros(c.m + 1)
        x_temp = c.X[j]
        x[0] = c.X[j]
        optic_length = 0
        c.x_i[j][0] = c.X[j]
        c.tg[j, 0] = 0  # -15.85E-3
        c.dPhi[j] = 2 * np.pi * np.sqrt(0.19 * 0.19) / 5320 * 1E10  # -15.85E-3
        c.dPhi_init[j] = c.dPhi[j]
        for i in range(1, c.m + 1):
            tgA[i] = 1 / n_curved(x_temp, i * c.dz + c.dz / 2) * (
                        n_x_curved(x_temp, i * c.dz) * c.dz - n_z_curved(x_temp, i * c.dz) * c.dz * c.tg[j, i])
            c.tg[j, i] = tgA[i] + c.tg[j, i - 1]
            x_temp = x[i - 1] + c.dz / 2. * c.tg[j, i]
            x[i] = x[i - 1] + c.dz * c.tg[j, i]
            optic_length += n_curved(x_temp, i * c.dz) * np.sqrt(c.dz * c.dz + (x[i] - x[i - 1]) * (x[i] - x[i - 1]))
            c.x_i[j, i], c.z_i[j, i] = x[i], -i * c.dz
        c.dPhi[j] += 2 * np.pi * optic_length / 5320 * 1E10
        file.write(str(c.x_i[j][c.m]) + ' ' + str(c.dPhi[j]) + '\n')
        c.X[j] = c.x_i[j][c.m]
        #plt.plot(c.x_i[j], c.z_i[j])
   # plt.savefig('5.eps', format='eps')
