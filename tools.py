import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate


def integrate_log(f, a, b, args=None, epsrel=1e-4):
    if isinstance(f, np.ndarray):
        x_array = np.linspace(np.log(a), np.log(b), len(f))
        return integrate.simps(f * np.exp(x_array), x_array)
    if args is None:
        lna = np.log(a)
        lnb = np.log(b)

        def logf(x):
            return f(np.exp(x))*np.exp(x)

        return integrate.quad(logf, lna, lnb, epsrel=epsrel)[0]
    else:
        lna = np.log(a)
        lnb = np.log(b)

        def logf(x, *args):
            return f(np.exp(x), *args)*np.exp(x)

        return integrate.quad(logf, lna, lnb, epsrel=epsrel, args=args, limit=2)[0]


def convolve_log(f1, f2, x_grid, y_grid=None):
    if y_grid is None:
        y_grid = x_grid
    f1_func = interpolate.interp1d(x_grid, f1)
    f2_func = interpolate.interp1d(y_grid, f2)

    def f12_func(x, y):
        ymx = (y-x)*(y - x > x_grid[0]) + x_grid[0] * (y - x < x_grid[0])
        return (y - x > x_grid[0]) * f1_func(x[:]) * f2_func(ymx)  # f1_func(x)*f2_func(y) #(x < y-x_grid[0])*f1_func(x)*f2_func((y)*(x < y-x_grid[0]))
    dx = np.zeros(len(x_grid))
    dx[1:] = (x_grid[1:] - x_grid[:-1]) / 2
    dx[:-1] += (x_grid[1:] - x_grid[:-1]) / 2
    xx, yy = np.meshgrid(x_grid, x_grid)
    convolution = np.sum(f12_func(xx, yy)*dx, axis=1)  # integrate.simps(f12_func(xx, yy),x_grid)
    return convolution


def azimutal_average_3d(inmap, x, y, z, dr, x0=0, y0=0, z0=0):
    x_ind, y_ind, z_ind = np.indices(inmap.shape)

    r = np.sqrt((x[x_ind] - x0) ** 2
                + (y[y_ind] - y0) ** 2
                + (z[z_ind] - z0) ** 2)

    # np.hypot(x[x_ind] - x0, y[y_ind] - y0, z[z_ind] - z0)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] / dr
    map_sorted = inmap.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    delta_r = r_int[1:] - r_int[:-1]  # Assumes all dr intervals represented

    rind = np.where(delta_r)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(map_sorted, dtype=float)
    sum_rbin = csim[rind[1:]] - csim[rind[:-1]]

    return sum_rbin / nr, (r_int[rind[1:]] + 0.5) * dr  # average value of function in each radial bin of length dr


def generate_poisson_map(lambda_map):
    output_map = np.zeros_like(lambda_map)
    output_map[np.where(lambda_map < 0)] = 0
    output_map[np.where(lambda_map >= 0)] = np.random.poisson(lambda_map[np.where(lambda_map >= 0)])
    return output_map


def get_inv_cdf(func, edges, log=True, args=None):
    if log:
        n_cmf = 5000
        cdf = np.zeros(n_cmf)
        log_x = np.linspace(np.log10(edges[0]), np.log10(edges[1]), n_cmf)
        for i in range(n_cmf):
            cdf[i] = integrate.quad(func, 10 ** log_x[0], 10 ** log_x[i], epsrel=1e-9, args=args)[0]

        norm = cdf[-1]
        if norm < 0.99 or norm > 1.01:
            print "Pdf not exactly normalized on this interval, renormalizing. Norm = ", norm

        cdf /= norm

        inv_cdf_func = interpolate.interp1d(cdf, 10 ** log_x)
    else:
        n_cmf = 5000
        cdf = np.zeros(n_cmf)
        x = np.linspace(edges[0], edges[1], n_cmf)
        for i in range(n_cmf):
            cdf[i] = integrate.quad(func, x[0], x[i], epsrel=1e-9, args=args)[0]

        norm = cdf[-1]
        if norm < 0.99 or norm > 1.01:
            print "Pdf not exactly normalized on this interval, renormalizing. Norm = ", norm
        cdf /= norm

        inv_cdf_func = interpolate.interp1d(cdf, x)

    return inv_cdf_func


def vid_from_cube(cubename, temp_range=None):
    if temp_range is None:
        n = 100
        temp_range = np.logspace(-9, -4, n + 1)
    # else:
    #     n = len(temp_range) - 1

    cube = np.load(cubename)
    my_hist = np.histogram((cube.f.t.flatten() + 1e-12) * 1e-6, bins=temp_range)[0]
    n_vox = len(cube.f.t.flatten())
    dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
    x = (temp_range[1:] + temp_range[:-1]) / 2

    return my_hist / dtemp_times_n_vox, x
