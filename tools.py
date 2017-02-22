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
