import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
scaling = 1
NS_density = open ("/Users/mikhail/Diffraction/NS_density.dat")
x_den = np.array([])


def n(x):
    b = 0.000014*scaling
    return n_polyn(-1.39*1E-5*scaling) if x <= -b else n_polyn(b) if x >= b else n_polyn(x)


def n_dash(x):
    b = 1.39*1E-5*scaling
    return n_dash_polyn(-1.39*1E-5*scaling) if x <= -b else n_dash_polyn(1.39*1E-5*scaling) if x >= b else n_dash_polyn(x)


def n_time(x):
    return (n(x)+n(x + 4E6))/2


def n_dash_time(x):
    return (n_dash(x)+n_dash(x + 4E6))/2


m = 1000
n_polyn = np.array([])
n_polyn_dash = np.array([])
d_Phi = np.array([])
with open('NS_density.dat') as f:
    for line in f:
        temp_x, temp_y = [float(x) for x in line.split()] 
        x_den = np.append(x_den,scaling*temp_x)
        n_polyn = np.append(n_polyn,0.039*temp_y)

Ta = 293.0
R = 8.31144598/0.029
Width = 0.048
dz = Width / m

n_polyn = 1 + 2.27*np.power(10., -4)*n_polyn
n_dash_polyn = np.diff(n_polyn)/np.diff(x_den)
n_polyn = inter.interp1d(x_den,n_polyn)
n_dash_polyn = inter.interp1d(x_den[1:],n_dash_polyn)

t = np.arange(-0.00002, 0.00002, 0.0000001)
res = np.array([])

boundary = 1E-5*scaling
step = 2.5E-7*scaling
X = np.arange(-boundary,boundary+step, step)#125E-8, /2, 50-7

x_i = np.zeros((np.size(X),m+1))
z_i = np.zeros((np.size(X),m+1))
dPhi = np.zeros(np.size(X))
tg = np.zeros((np.size(X),m+1))
diversity = 0.20
for j,x0 in enumerate(X):
    print(str(j)+'/'+str(np.size(X)-1))
    x = np.zeros(m+1)
    tgA = np.zeros(m+1)
    x_temp = x0 
    x[0] = x0
    l = 0
    tg[j,0] = x0/diversity
    for i in range(1,m+1):
        tgA[i] = 1/n_time(x_temp) * n_dash_time(x_temp) * dz
        tg[j,i] = tgA[i] + tg[j,i-1]
        x_temp = x[i-1] + dz/2. * tg[j,i]
        x[i] = x[i-1] + dz * tg[j,i]
        l += n_time(x_temp) * np.sqrt(dz * dz + (x[i] - x[i-1]) * (x[i] - x[i-1]))
        x_i[j,i], z_i[j,i] = x[i],  -i*dz
    dPhi[j] = 2*np.pi*l/5320*1E10

fig, ax = plt.subplots()
for j,x0 in enumerate(X):
    ax.plot(x_i[j], z_i[j])
ax.set(title='Light trajectories')
fig.savefig("lt_range"+str(boundary)+"step"+str(step)+"scaling"+str(scaling)+"diversity"+str(diversity)+"blurry.png")
coordinate_dPhi_file = open("x_dPhi_range"+str(boundary)+"step"+str(step)+"scaling"+str(scaling)+"diversity"+str(diversity)+"blurry_x.txt", "w")
for j,x0 in enumerate(X):
    coordinate_dPhi_file.write(str(x0) + ' ' + str(x_i[j][m]) + ' ' + str(dPhi[j]) + '\n')
