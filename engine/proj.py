"""The stage projection, shared by every module: an oblique three-quarter view from slightly above.

stage_x = x, stage_y = Y0 + z * FS - y (x along the screen, z into depth, y height). The stage module sets Y0 and FS
when the world is built (its PROJ); everything that draws in stage coordinates calls floor_y.
"""
P = {'Y0': 239.0, 'FS': 0.6}


def set_projection(y0, fs):
    P['Y0'], P['FS'] = float(y0), float(fs)


def floor_y(z):
    """The stage y of the ground (height 0) at depth z; works on numbers and numpy arrays."""
    return P['Y0'] + z * P['FS']


def proj(x, z, y=0.0):
    return x, P['Y0'] + z * P['FS'] - y
