import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import constants as c

NS_density = open(c.density_path)

x_den = np.array([])


def n(x):
    b = 0.000014 * c.scaling
    return n_polyn(-1.39 * 1E-5 * c.scaling) if x <= -b else n_polyn(b) if x >= b else n_polyn(x)


def n_dash(x):
    b = 1.39 * 1E-5 * c.scaling
    return n_dash_polyn(-1.39 * 1E-5 * c.scaling) if x <= -b else n_dash_polyn(1.39 * 1E-5 * c.scaling) if x >= b \
        else n_dash_polyn(x)


def n_time(x):
    return (n(x) + n(x + 4E6)) / 2


def n_dash_time(x):
    return (n_dash(x) + n_dash(x + 4E6)) / 2


def r(i1, x1):
    return np.sqrt(c.z0 * c.z0 + (X[i1] - x1) * (X[i1] - x1))


@njit(parallel=True)
def Vectorizing_sum(v_Re, v_Im):
    Re = v_Re.sum()
    Im = v_Im.sum()
    return Re * Re + Im * Im


def Results(x):
    for d, v0 in enumerate(vectorize_sum_Re):
        vectorize_sum_Re[d] = A(X[d]) * np.power(c.z0 / r(d, x), 0.5) * np.cos(2 * np.pi / c.lam * (r(d, x) - c.z0) + dPhi[d])
        vectorize_sum_Im[d] = A(X[d]) * np.power(c.z0 / r(d, x), 0.5) * np.sin(2 * np.pi / c.lam * (r(d, x) - c.z0) + dPhi[d])
    return Vectorizing_sum(vectorize_sum_Re, vectorize_sum_Im)


def A(variable):
    if (variable + c.Object / 2.0) <= 10.0 * np.power(10.0, -3):
        return np.power(np.sin(np.pi / 2 / (10 * np.power(10.,-3)) * (variable + c.Object / 2)), 2)
    else:
        if (variable + c.Object / 2.0) >= (c.Object - 10.0 * np.power(10.0, -3)):
            return np.power(np.sin(np.pi / 2 / (10 * np.power(10., -3)) * (variable -  c.Object / 2)), 2)
        else:
            return 1


data_dir = Path('data')
if not data_dir.exists():
    data_dir.mkdir()
    path = 'data/result_1/'
    Path(path).mkdir()
    base = open("data/log.txt", "w")
    base.write('result_1 ' + str(datetime.now()) + ' ' + '\n')

else:
    count = len(list(data_dir.glob('result_*')))
    current_dir = data_dir / f'result_{count + 1}'
    current_dir.mkdir()
    path = str(current_dir) + '/'
    base = open("data/log.txt", "a")
    base.write('result_'+str(count+1)+' '+str(datetime.now())+ ' '+'\n')

print('Results and debug files at:', path)
debug = open(path+'debug.txt', "w")

n_polyn = np.array([])
n_polyn_dash = np.array([])
d_Phi = np.array([])
with open('NS_density.dat') as f:
    for line in f:
        temp_x, temp_y = [float(x) for x in line.split()]
        x_den = np.append(x_den, c.scaling * temp_x)
        n_polyn = np.append(n_polyn, 0.039 * temp_y)

# Vertical step:
dz = c.Width / c.m

n_polyn = 1 + 2.27 * np.power(10., -4) * n_polyn
n_dash_polyn = np.diff(n_polyn) / np.diff(x_den)
n_polyn = inter.interp1d(x_den, n_polyn)
n_dash_polyn = inter.interp1d(x_den[1:], n_dash_polyn)

X = np.arange(-c.Object / 2., c.Object / 2.0, c.step)
# this is all the c.Object lightning up, with both JK_left and JK_right
boundary_index = int((c.Object / 2.0 - c.boundary) / c.step)
zero_left = np.zeros(boundary_index)
zero_right = np.zeros(boundary_index)
x_i_left = np.arange(-c.Object / 2., c.boundary + c.step, c.step)
x_i_right = np.arange(c.boundary, c.Object / 2.0 + c.step, c.step)
x_i = np.zeros((np.size(X), c.m + 1))
z_i = np.zeros((np.size(X), c.m + 1))
dPhi = np.zeros(np.size(X))
tg = np.zeros((np.size(X), c.m + 1))

coordinate_dPhi_file = open(path + 'results.dat', "w")

print('Solving shockwave passthroughs')
for j in tqdm(np.arange(boundary_index, np.size(X)-boundary_index + 1, 1)):
    x = np.zeros(c.m + 1)
    tgA = np.zeros(c.m + 1)
    x_temp = X[j]
    x[0] = X[j]
    optic_length = 0
    tg[j, 0] = (j-boundary_index) / c.diversity
    for i in range(1, c.m + 1):
        tgA[i] = 1 / n_time(x_temp) * n_dash_time(x_temp) * dz
        tg[j, i] = tgA[i] + tg[j, i - 1]
        x_temp = x[i - 1] + dz / 2. * tg[j, i]
        x[i] = x[i - 1] + dz * tg[j, i]
        optic_length += n_time(x_temp) * np.sqrt(dz * dz + (x[i] - x[i - 1]) * (x[i] - x[i - 1]))
        x_i[j, i], z_i[j, i] = x[i], -i * dz
    dPhi[j] = 2 * np.pi * optic_length / 5320 * 1E10
    coordinate_dPhi_file.write(str(X[j]) + ' ' + str(x_i[j][c.m]) + ' ' + str(dPhi[j]) + '\n')


fig, ax = plt.subplots()
for j, x0 in enumerate(X):
    ax.plot(x_i[j], z_i[j])
ax.set(title='Light trajectories')
fig.savefig(path + 'light_trajectories.png')

#
#  Mathematica code rewritten:
#
dPhi = np.append(zero_left, dPhi)
dPhi = np.append(dPhi, zero_right)
x_i = np.append(x_i_left, x_i)
x_i = np.append(x_i, x_i_right)

vectorize_sum_Re = np.zeros(np.size(X))
vectorize_sum_Im = np.zeros(np.size(X))

intensity_file = open(path + 'intensity.dat', "w")
xp_range = np.arange(-6E-3, 6E-3, 0.02E-3)
intensity_dat = np.array([])
print('Summation process')
for xp in tqdm(xp_range):
    intensity = Results(xp)
    intensity_dat = np.append(intensity_dat, intensity)
    intensity_file.write(str(xp) + ' ' + str(intensity)+'\n')
fig2, ax2 = plt.subplots()
ax2.plot(xp_range, intensity_dat)
ax2.set(title='Diffraction')
fig2.savefig(path + 'Diffraction.png')

A_data = np.copy(X)
for j,x0 in enumerate(X):
    A_data[j] = A(x0)
fig3,ax3 = plt.subplots()
ax3.plot(X, A_data)
fig3.savefig(path+'A.png')
base.close()
coordinate_dPhi_file.close()
intensity_file.close()