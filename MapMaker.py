import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy.interpolate import RegularGridInterpolator
import scipy.integrate as integrate
import ConfigParser
import os

import tools
import VID
import ast


# Class for generating maps used for vid-analysis. Does not take into account the change in redshift,
# but assumes a rectangular grid of voxels. Note that x, y and z are arrays with the voxel centers,
# not the voxel centers.
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
        self.fx = fft.fftshift(fft.fftfreq(self.n_x, self.dx))
        self.fy = fft.fftshift(fft.fftfreq(self.n_y, self.dy))
        self.fz = fft.fftshift(fft.fftfreq(self.n_z, self.dz))

    def create_map(self, number_of_maps=1, power_spectrum_function=None, log_normal=False, sigma_g=1.0, *args, **kwargs):
        if power_spectrum_function is None:
            ps_values = sigma_g ** 2 * self.vox_vol
        else:
            # ps_values = P(k) = volume < |d_k|^2 >
            ps_values = self.normalize_power_spectrum(sigma_g, power_spectrum_function, *args, **kwargs)
        field = np.random.randn(number_of_maps, self.n_x, self.n_y, self.n_z, 2)

        fft_field = np.zeros((number_of_maps, self.n_x, self.n_y, self.n_z), dtype=complex)

        # This normalization ensures that the real part and the imaginary of map has right power spectrum, but not
        # the full complex map, which will have a factor 2 too large power spectrum.
        fft_field[:] = (field[:, :, :, :, 0] + 1j * field[:, :, :, :, 1]) \
                       * np.sqrt(ps_values[None] / self.volume)

        # Multiply by n_x * n_y * n_z, because inverse function in python divides by N, but that is
        # not in our convention for cosmology
        out_map = np.real(np.fft.ifftn(fft_field, axes=(1, 2, 3))) * self.n_x * self.n_y * self.n_z
        print "sigma = ", out_map.flatten().std()
        print "sigma_g =", sigma_g
        if log_normal:
            out_map = np.exp(out_map - sigma_g ** 2 / 2.0) - 1.0
        return out_map

    # Normalizes the power spectrum such that the voxel variance, sigma_g^2, is correct.
    def normalize_power_spectrum(self, sigma, ps_func, *args, **kwargs):
        w = np.zeros((self.n_x, self.n_y, self.n_z))
        w[0, 0, 0] = 1
        w_fft = fft.fftn(w)
        # Multiply by 2 pi here to transform frequencies to k, k = 2 pi f, f_l = l / L, l = 0, 1, ... n-1
        p_k = ps_func(2 * np.pi * np.abs(np.sqrt(self.fx[:, None, None] ** 2
                                                       + self.fy[None, :, None] ** 2
                                                       + self.fz[None, None, :] ** 2)), *args, **kwargs) / self.volume
        norm_factor = np.sum(np.abs(w_fft) ** 2 * p_k)
        return p_k / (norm_factor / sigma ** 2) * self.volume

    # Does not take into account changes with redshift properly, but that would be complicated!
    def calculate_power_spec_3d(self, in_map):
        # just something to get reasonable values for dk, not very good
        dk = 1 * np.sqrt(np.sqrt(self.dx * self.dy)) / np.sqrt(self.volume)

        fft_map = fft.fftn(in_map) / (self.n_x * self.n_y * self.n_z)
        fft_map = fft.fftshift(fft_map)
        ps = np.abs(fft_map) ** 2 * self.volume
        return tools.angular_average_3d(ps, self.fx, self.fy, self.fz, dk)

    # Generates a custom cube from a luminosity function and a power spectrum.
    def generate_cube(self, lum_func=None, power_spectrum=None, sigma_g=1.0,
                      parameterfile='parameters.ini', cubename='custom_cube',
                      lum_args=None, ps_args=None, save_cube=True):
        if power_spectrum is None:
            power_spectrum = tools.power_law_ps
        if ps_args is None:
            ps_args = {}

        if lum_func is None:
            lum_func = VID.VoxelIntensityDistribution.default_luminosity_function
            if lum_args is None:
                config = ConfigParser.ConfigParser()
                config.read(parameterfile)
                lum_args = np.array(ast.literal_eval(config.get("Luminosity", "arguments"))).astype(float)

        vid = VID.VoxelIntensityDistribution(parameterfile)

        lum_func_cdf = tools.get_inv_cdf(lum_func, (1e-1, 1e9), log=True, args=lum_args)

        my_map = self.create_map(power_spectrum_function=power_spectrum, number_of_maps=1,
                                 log_normal=True, sigma_g=sigma_g, **ps_args)[0]

        print r'Voxel volume in (Mpc/h)^3:', vid.vol_vox
        density = integrate.quad(lum_func, 1e0, 1e8, epsrel=1e-9, args=lum_args)[0]
        sources_per_voxel = density * vid.vol_vox

        print r'Average number of sources per voxel (theory):', sources_per_voxel

        mean_map = sources_per_voxel * (1.0 + my_map)

        source_map = tools.generate_poisson_map(mean_map)

        print "Maximum number of sources in voxel:", source_map.flatten().max()

        print "Mean source density in map:", mean_map.flatten().mean()

        lum_map = np.zeros_like(source_map)

        for i in range(self.n_x):
            for j in range(self.n_y):
                for k in range(self.n_z):
                    lum_map[i, j, k] = lum_func_cdf(np.random.uniform(size=int(source_map[i, j, k]))).sum()
        temp_map = vid.x_lt / vid.vol_vox * lum_map * 1e6
        if save_cube:
            if not os.path.isdir('cubes/'):
                os.makedirs('cubes/')
            np.savez("cubes/" + cubename, t=temp_map, x=self.x, y=self.y, z=self.z)
        return temp_map, self.x, self.y, self.z
