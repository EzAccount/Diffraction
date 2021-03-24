import constants as c
import numpy as np


def n(x, z):
    return c.n_polyn(-c.wave_width * c.scaling) if x <= -c.wave_width * c.scaling else c.n_polyn(
        c.wave_width* c.scaling) if x >= c.wave_width* c.scaling else c.n_polyn(x)


def n_dash(x, z):
    return c.n_polyn_dash(-c.wave_width * c.scaling) if x <= -c.wave_width * c.scaling else c.n_polyn_dash(
        c.wave_width * c.scaling) if x >= c.wave_width * c.scaling \
        else c.n_polyn_dash(x)


def n_curved(x, z):
    return n(x - c.curve * np.cos(np.pi * (z - c.Width / 2) / c.Width), z)


def n_z_curved(x, z):
    return (n_curved(x, z + c.dz / 2) - n_curved(x, z - c.dz / 2)) / c.dz


def n_x_curved(x, z):
    return n_dash(x - c.curve * np.cos(np.pi * (z - c.Width / 2) / c.Width), z)
