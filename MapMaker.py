import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.interpolate import RegularGridInterpolator
import scipy.integrate as integrate


class MapMaker:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.n_x = len(x) - 1
        self.n_y = len(y) - 1
        self.n_z = len(z) - 1
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
        self.volume = ((x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0]))
        self.vox_vol = self.volume / (self.n_x * self.n_y * self.n_z)

    def create_map(self, number_of_maps=1, power_spectrum_function=None, log_normal=False, sigma_g=1, *args, **kwargs):
        if power_spectrum_function is None:
            ps_values = sigma_g ** 2 * self.vox_vol
        else:
            # ps_values = P(k) = volume < |d_k|^2 >
            ps_values = self.normalize_power_spectrum(sigma_g, power_spectrum_function, *args, **kwargs)

        field = np.random.randn(number_of_maps, self.n_x, self.n_y, self.n_z, 2)

        fft_field = np.zeros((number_of_maps, self.n_x, self.n_y, self.n_z), dtype=complex)
        fft_field[:] = (field[:, :, :, :, 0] + 1j * field[:, :, :, :, 1]) \
                       * np.sqrt(ps_values[None] / self.volume)
        # Multiply by n_x * n_y * n_z, because inverse function in python divides by N, but that is
        # not in my cosmological convention
        out_map = np.real(np.fft.ifftn(fft_field, axes=(1, 2, 3))) * self.n_x * self.n_y * self.n_z
        print "sigma = ", out_map.flatten().std()
        print "sigma_g =", sigma_g
        if log_normal:
            out_map = np.exp(out_map - sigma_g ** 2 / 2.0) - 1.0
        return out_map

    # def create_map(self, number_of_maps=1, power_spectrum_function=None, log_normal=False, sigma_g=1):
    #
    #     if power_spectrum_function is None:
    #         ps_values = sigma_g ** 2 * self.vox_vol
    #     else:
    #         # ps_values = P(k) = volume < |d_k|^2 >
    #         ps_values = self.normalize_power_spectrum(sigma_g, power_spectrum_function)
    #
    #     field = np.random.randn(self.n_x, self.n_y, self.n_z, 2)
    #
    #     fft_field = np.zeros((self.n_x, self.n_y, self.n_z), dtype=complex)
    #     fft_field[:] = (field[:, :, :, 0] + 1j * field[:, :, :, 1]) \
    #                    * np.sqrt(ps_values / self.volume)
    #     # Multiply by n_x * n_y * n_z, because inverse function in python divides by N, but that is
    #     # not in my cosmological convention
    #     out_map = np.real(np.fft.ifftn(fft_field)) * self.n_x * self.n_y * self.n_z
    #     print "sigma = ", out_map.flatten().std()
    #     print "sigma_g =", sigma_g
    #     if log_normal:
    #         out_map = np.exp(out_map - sigma_g ** 2 / 2.0) - 1.0
    #     return out_map

    def normalize_power_spectrum(self, sigma, ps_func, *args, **kwargs):

        w = np.zeros((self.n_x, self.n_y, self.n_z))
        w[0, 0, 0] = 1
        # This should be w[0, 0, 0] = self.volume / self.vox_vol,
        # but for numerical stability we insert this factor in the normalization integral later
        w_fft = fft.fftn(w)
        fx, fy, fz = (fft.fftshift(fft.fftfreq(self.n_x, self.dx)), fft.fftshift(fft.fftfreq(self.n_y, self.dy)),
                      fft.fftshift(fft.fftfreq(self.n_z, self.dz)))
        # Multiply by 2 pi here to transform frequencies to k, k = 2 pi f, f_l = l / L, l = 0, 1, ... n-1
        p_k = ps_func(2 * np.pi * np.abs(np.sqrt(fx[:, None, None] ** 2
                                                       + fy[None, :, None] ** 2
                                                       + fz[None, None, :] ** 2)), *args, **kwargs) / self.volume
        integral = np.sum(np.abs(w_fft) ** 2 * p_k)
        return p_k / (integral / sigma ** 2) * self.volume
