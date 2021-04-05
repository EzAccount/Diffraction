from numba import njit
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from n_functions import *
import light_trajectories as lt


def r(i1, x1):
    return np.sqrt(c.z0 * c.z0 + (c.X[i1] - x1) * (c.X[i1] - x1))


@njit(parallel=True)
def sum_vectorization(v_re, v_im):
    Re = v_re.sum()
    Im = v_im.sum()
    return Re * Re + Im * Im


def Results(x):
    for d, v0 in enumerate(vectorized_sum_Re):
        vectorized_sum_Re[d] = A(c.X[d]) * np.power(c.z0 / r(d, x), 0.5) * np.cos(
            2 * np.pi / c.lam * (r(d, x) - c.z0) + c.dPhi[d])
        vectorized_sum_Im[d] = A(c.X[d]) * np.power(c.z0 / r(d, x), 0.5) * np.sin(
            2 * np.pi / c.lam * (r(d, x) - c.z0) + c.dPhi[d])
    return sum_vectorization(vectorized_sum_Re, vectorized_sum_Im)


def A(variable):
    if (variable + c.Object / 2.0) <= 10.0 * np.power(10.0, -3):
        return np.power(np.sin(np.pi / 2 / (10 * np.power(10., -3)) * (variable + c.Object / 2)), 2)
    else:
        if (variable + c.Object / 2.0) >= (c.Object - 10.0 * np.power(10.0, -3)):
            return np.power(np.sin(np.pi / 2 / (10 * np.power(10., -3)) * (variable - c.Object / 2)), 2)
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
    base.write('result_' + str(count + 1) + ' ' + str(datetime.now()) + ' ' + '\n')

print('Results and debug files at:', path)
debug = open(path + 'debug.txt', "w")
profile = open(path + 'profile.txt', "w")
dPhi_file = open(path + 'c.dPhi.txt', "w")
dPhi_init_file = open(path + 'dPhi_init.txt', "w")

i = 0
with open(c.density_path, encoding='utf-8-sig') as f:  # , encoding='utf-16'
    for line in f:
        i = i + 1
        temp_x, temp_y = [float(x) for x in line.split( )]
        c.x_den = np.append(c.x_den, c.scaling * temp_x)
        c.n_polyn = np.append(c.n_polyn, 0.039 * temp_y)

# Vertical step:


c.n_polyn = 1 + 2.27 * np.power(10., -4) * c.n_polyn
c.n_polyn_dash = np.diff(c.n_polyn) / np.diff(c.x_den)
c.n_polyn = inter.interp1d(c.x_den, c.n_polyn)
c.n_polyn_dash = inter.interp1d(c.x_den[1:], c.n_polyn_dash)

dPhi_init = np.zeros(np.size(c.X))
# this is all the c.Object lightning up, with both JK_left and JK_right
boundary_index = int((c.Object / 2.0 - c.boundary) / c.step)

coordinate_dPhi_file = open(path + 'results.dat', "w")

fig0 = plt.figure()
ax0 = plt.axes(projection='3d')
ax0.set_xlabel('c.X')
ax0.set_ylabel('Z')

x = np.arange(-2 * c.wave_width, 2 * c.wave_width, c.step / 2)
z = np.arange(0., c.Object, c.dz)
Xp, Zp = np.meshgrid(x, z)
N = np.zeros((np.size(x), np.size(z)))
for i, x in enumerate(x):
    for j, y in enumerate(z):
        N[i][j] = n_curved(x, y)
ax0.scatter(Xp, Zp, N.transpose())
#plt.show()
#fig0.savefig(path + 'lanes.png')

reference_flag = input("Reference light?[Y/n]")
if reference_flag == 'n':
    trajectory = open('refs/' + 'trajectory.txt', "w")
    lt.Solve(trajectory)
else:
    trajectory = open('refs/' + input("Filename:"), "r")
    for j, line in enumerate(trajectory):
        c.X[j], c.dPhi[j] = line.split()

vectorized_sum_Re = np.zeros(np.size(c.X))
vectorized_sum_Im = np.zeros(np.size(c.X))

intensity_file = open(path + 'intensity.dat', "w")
xp_range = np.arange(-0.06, 0.06, 0.00001)  # (-6E-2, 6E-2, 0.2E-3)
intensity_dat = np.array([])
print('Summation process')

for xp in tqdm(xp_range):
    intensity = Results(xp)
    intensity_dat = np.append(intensity_dat, intensity)
    intensity_file.write(str(xp) + ' ' + str(intensity) + '\n')

fig2, ax2 = plt.subplots()
ax2.plot(xp_range, intensity_dat)
ax2.set(title='Diffraction')
#fig2.savefig(path + 'Diffraction.png')

A_data = np.copy(c.X)
for j, x0 in enumerate(c.X):
    A_data[j] = A(x0)
fig3, ax3 = plt.subplots()
ax3.plot(c.X, A_data)
#fig3.savefig(path + 'A.png')
base.close()
coordinate_dPhi_file.close()
intensity_file.close()
