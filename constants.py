#  Physical constants and object properties:
import numpy as np
Ta = 293.0
R = 8.31144598 / 0.029
Width = 0.048
Object = 5E-2  # 5E-2
lam = 5320 * np.power(10., -10)
z0 = 800 * np.power(10., -3)
scaling = 1
boundary = 20E-6 * scaling
step = 1E-6

m = 1000
diversity = 1
density_path = "NS_density_new_2p2.dat"
curve = 0
wave_width = 0.000013
x_den = np.array([])

X = np.arange(-Object/2, Object/2, step)  # -Object / 2., Object / 2.0, step
dPhi_init = np.zeros(np.size(X))
n_polyn = np.array([])
n_polyn_dash = np.array([])
x_i = np.zeros((np.size(X), m + 1))
z_i = np.zeros((np.size(X), m + 1))
dPhi = np.zeros(np.size(X))
tg = np.zeros((np.size(X), m + 1))
dz = Width/m
