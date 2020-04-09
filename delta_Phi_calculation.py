import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from pathlib import Path
from datetime import datetime

scaling = 1
NS_density = open("/Users/mikhail/Diffraction/NS_density.dat")
x_den = np.array([])


def n(x):
    b = 0.000014 * scaling
    return n_polyn(-1.39 * 1E-5 * scaling) if x <= -b else n_polyn(b) if x >= b else n_polyn(x)


def n_dash(x):
    b = 1.39 * 1E-5 * scaling
    return n_dash_polyn(-1.39 * 1E-5 * scaling) if x <= -b else n_dash_polyn(1.39 * 1E-5 * scaling) if x >= b \
        else n_dash_polyn(x)


def n_time(x):
    return (n(x) + n(x + 4E6)) / 2


def n_dash_time(x):
    return (n_dash(x) + n_dash(x + 4E6)) / 2


def r(i1, x1):
    return np.sqrt(z0 * z0 + (X[i1] - x1) * (X[i1] - x1))


def Results(x):
    for v0, d in enumerate(vectorize_sum_Re):
        vectorize_sum_Re[d] = A(X[d]) * np.power(z0 / r(d, x), 0.5) * np.cos(
            2 * np.pi / lam * (r(d, x) - z0) + dPhi[d])
        vectorize_sum_Im[d] = A(X[d]) * np.power(z0 / r(d, x), 0.5) * np.sin(
            2 * np.pi / lam * (r(d, x) - z0) + dPhi[d])
    Re = vectorize_sum_Re.sum()
    Im = vectorize_sum_Im.sum()
    return Re * Re + Im * Im


def A(variable):
    if (variable - Object / 2.0) <= 10.0 * np.power(10, -3):
        return np.power(np.sin(np.pi / 2 / (10 * 10 ^ (-3)) * (variable - Object / 2)), 2)
    else:
        if (variable - Object / 2.0) >= (Object - 10.0 * np.power(10, -3)):
            return np.power(np.sin(np.pi / 2 / (10 * 10 ^ (-3)) * (variable - 3 * Object / 2)), 2)
        else:
            return 1


data_dir = Path('data')
if not data_dir.exist():
    data_dir.makedir()
    path = 'data/01/'
    Path(path).makedir()
    base = open("data/log.txt", "w")
    base.write('01 ' + datetime.ctime() + ' ')

else:
    count = len(data_dir.glob('result_*'))
    current_dir = data_dir / f'result_{count + 1}'
    current_dir.makedir()
    path = str(current_dir)

m = 1000
n_polyn = np.array([])
n_polyn_dash = np.array([])
d_Phi = np.array([])
with open('NS_density.dat') as f:
    for line in f:
        temp_x, temp_y = [float(x) for x in line.split()]
        x_den = np.append(x_den, scaling * temp_x)
        n_polyn = np.append(n_polyn, 0.039 * temp_y)

# Physical constants and object properties:
Ta = 293.0
R = 8.31144598 / 0.029
Width = 0.048
Object = 5E-2
lam = 5320 * np.power(10, -10)
z0 = 800 * 10 ^ (-3)
boundary = 1E-5 * scaling
step = 2.5E-7 * scaling

# Vertical step:
dz = Width / m

n_polyn = 1 + 2.27 * np.power(10., -4) * n_polyn
n_dash_polyn = np.diff(n_polyn) / np.diff(x_den)
n_polyn = inter.interp1d(x_den, n_polyn)
n_dash_polyn = inter.interp1d(x_den[1:], n_dash_polyn)

X = np.arange(-Object / 2., Object / 2.0, step)  # this is all the object lightning up, with both JK_left and JK_right
boundary_index = int((Object / 2.0 - boundary) / step)
zero_left = np.zeros(boundary_index)
zero_right = np.zeros(boundary_index)
x_i_left = np.arange(-Object / 2., boundary + step, step)
x_i_right = np.arange(boundary, Object / 2.0 + step, step)
x_i = np.zeros((np.size(X), m + 1))
z_i = np.zeros((np.size(X), m + 1))
dPhi = np.zeros(np.size(X))
tg = np.zeros((np.size(X), m + 1))
diversity = 0.20
coordinate_dPhi_file = open(path + 'results.dat', "w")

for j in np.range(-boundary_index, boundary_index + 1, 1):
    print(str(j) + '/' + str(np.size(X) - 1))
    x = np.zeros(m + 1)
    tgA = np.zeros(m + 1)
    x_temp = X[j]
    x[0] = X[j]
    optic_length = 0
    tg[j, 0] = j / diversity
    for i in range(1, m + 1):
        tgA[i] = 1 / n_time(x_temp) * n_dash_time(x_temp) * dz
        tg[j, i] = tgA[i] + tg[j, i - 1]
        x_temp = x[i - 1] + dz / 2. * tg[j, i]
        x[i] = x[i - 1] + dz * tg[j, i]
        optic_length += n_time(x_temp) * np.sqrt(dz * dz + (x[i] - x[i - 1]) * (x[i] - x[i - 1]))
        x_i[j, i], z_i[j, i] = x[i], -i * dz
    dPhi[j] = 2 * np.pi * optic_length / 5320 * 1E10
    coordinate_dPhi_file.write(str(X[j]) + ' ' + str(x_i[j][m]) + ' ' + str(dPhi[j]) + '\n')

dPhi = np.append(zero_left, dPhi)
dPhi = np.append(dPhi, zero_right)
x_i = np.append(x_i_left, x_i)
x_i = np.append(x_i, x_i_right)

fig, ax = plt.subplots()
for j, x0 in enumerate(X):
    ax.plot(x_i[j], z_i[j])
ax.set(title='Light trajectories')
fig.savefig(path + 'light_trajectories.png')

#
# Mathematica code rewritten:
#


vectorize_sum_Re = np.array(np.size(X))
vectorize_sum_Im = np.array(np.size(X))

intensity = open(path + 'intensity.dat', "w")
xp_range = np.arange(-6E-3, 6E-3, 0.02E-3)
for xp in xp_range:
    intensity.write(str(xp) + ' ' + str(Results(xp)))
