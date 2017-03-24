import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import numpy.fft as fft

import tools


class VoxelIntensityDistribution:
    nu_em = 115e9  # 115 GHz
    speed_of_light = 3e8
    k_boltz = 1.3807e-23  # J/K
    mpc = 3.086e22  # 1 mpc = 3.086e22 m
    solar_lum = 3.848e26  # W

    def __init__(self, lum_func, fiducial_values, fiducial_units, config):
        self.mode = config.get('Grid', 'mode')
        temp_max = float(config.get('Grid', 'temp_max'))
        self.n_temp = int(config.get('Grid', 'n_temp'))
        if self.mode == 'log':
            temp_min = float(config.get('Grid', 'temp_min'))
            self.temp_range = np.logspace(np.log10(temp_min), np.log10(temp_max), self.n_temp)
        elif self.mode == 'fft':
            self.temp_range = np.linspace(-temp_max, temp_max, self.n_temp)
            self.dtemp = self.temp_range[1] - self.temp_range[0]
            self.sigma_noise = float(config.get('Telescope', 'sigma_noise')) * 1e-6

        self.h = float(config.get('Cosmology', 'h'))
        self.omega_m = float(config.get('Cosmology', 'omega_m'))
        self.omega_l = float(config.get('Cosmology', 'omega_l'))
        self.n_max = int(config.get('Cosmology', 'n_max'))

        self.delta_nu = float(config.get('Telescope', 'delta_nu'))
        self.angular_res = float(config.get('Telescope', 'angular_res')) * (np.pi / (180 * 60))

        freq_max = float(config.get('Telescope', 'freq_max'))
        freq_min = float(config.get('Telescope', 'freq_min'))
        z_min = self.nu_em / freq_max - 1
        z_max = self.nu_em / freq_min - 1
        self.z_mid = (z_max + z_min) / 2

        self.fiducial_values = np.array(fiducial_values).astype(float)
        self.fiducial_units = np.array(fiducial_units).astype(float)
        self.lum_func = lum_func
        self.density = self.integral(self.lum_func, 1e1, 1e8, args=(self.fiducial_values, self.fiducial_units))
        self.vol_vox, self.x_lt = self.get_x_and_vol()
        print "Sources per voxel:", self.density * self.vol_vox

    # plt.loglog(self.temp_range,self.lum_func(self.temp_range*self.vol_vox/self.x_lt,self.fiducial_values,self.fiducial_units))
    # plt.ylabel(r'$\lum_func$',fontsize=20)
    # plt.show()


    def get_x_and_vol(self):
        comoving_distance = self.speed_of_light * integrate.quad(self.get_one_over_hubble, 0, self.z_mid)[0]
        vol_vox = 1.0 / (self.mpc ** 3) * (comoving_distance * self.angular_res) ** 2 * self.speed_of_light * \
                  self.get_one_over_hubble(self.z_mid) * (1 + self.z_mid) ** 2 * self.delta_nu / self.nu_em
        print "Volume of 1 voxel: %g [(mpc/h)^3]" % vol_vox
        x_lt = self.h ** 2 * self.solar_lum * 1.0 / (self.mpc ** 3) * self.speed_of_light ** 3 * (1 + self.z_mid) ** 2 / \
               (8 * np.pi * self.k_boltz * self.nu_em ** 3) * self.get_one_over_hubble(self.z_mid)
        return vol_vox, x_lt  # both given in (mpc/h)^3

    def get_one_over_hubble(self, z):
        hubble = 3.241e-18 * np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_l)  # in units of h/s
        return 1.0 / hubble

    def calculate_vid(self, local_values, x_array=None):
        density_local = self.integral(self.lum_func, 1e1, 1e8, args=(local_values, self.fiducial_units))
        prob_1 = np.zeros(self.n_temp)
        prob_1[np.where(self.temp_range > 0)] = self.vol_vox / (density_local * self.x_lt) \
                                                * self.lum_func(self.temp_range[np.where(self.temp_range > 0)]
                                                                * self.vol_vox / self.x_lt,
                                                                local_values, self.fiducial_units)
        prob_n = self.prob_of_temp_given_n(prob_1, self.n_max)
        prob_signal = np.zeros(self.n_temp)
        for i in range(1, self.n_max + 1):
            prob_signal += prob_n[i] * self.prob_of_n_sources(i, density_local * self.vol_vox, local_values[-1] ** 2)
        if self.mode == 'fft':
            prob_noise = 1.0 / np.sqrt(2 * np.pi * self.sigma_noise ** 2) \
                         * np.exp(- self.temp_range ** 2 / (2 * self.sigma_noise ** 2))
            prob_total = self.prob_of_n_sources(0, density_local * self.vol_vox, local_values[-1] ** 2) * prob_noise \
                         + fft.fftshift(np.abs(fft.irfft(fft.rfft(prob_signal) * fft.rfft(prob_noise))) * self.dtemp)
            # np.abs(fft.irfft(fft.rfft(prob_signal) * fft.rfft(prob_noise))) * self.dtemp
            # fft.fftshift(np.abs(fft.irfft(fft.rfft(prob_signal) * fft.rfft(prob_noise))) * self.dtemp)
        elif self.mode == 'log':
            prob_total = prob_signal
        if x_array is None:
            return prob_signal
        else:  # Interpolate in log-space ?
            p_func = interpolate.interp1d(self.temp_range,
                                          prob_signal)  # interpolate.splrep(self.temp_range, prob_signal, s=0)
            return p_func(x_array)  # interpolate.splev(x_array, p_func, der=0)

    def integral(self, func, x_low, x_high, args=None, epsrel=1e-6):
        if self.mode == 'log':
            if args is None:
                return tools.integrate_log(func, x_low, x_high, epsrel=epsrel)
            else:
                return tools.integrate_log(func, x_low, x_high, args=args, epsrel=epsrel)
        if self.mode == 'fft':
            if args is None:
                return integrate.quad(func, x_low, x_high, epsrel=epsrel)[0]
            else:
                return integrate.quad(func, x_low, x_high, args=args, epsrel=epsrel)[0]

    # Probability distribution of expected temperature from N sources given P1 = Probability distribution of expected temperature from 1 source
    # Returns an array of values for N = 1 to n_max

    def prob_of_temp_given_n(self, prob_1, n):
        if self.mode == 'fft':
            prob_1_fft = fft.rfft(prob_1)
            prob_n = np.zeros((n + 1, len(prob_1)))
            prob_1_fft_n = 1
            for i in range(1, n + 1):
                # prob_of_temp_given_n[i] = np.abs(fft.irfft(prob_1_fft**i))*self.dtemp**(i-1)
                prob_1_fft_n *= prob_1_fft
                # Abs, because fft +ifft can make ~zero into sligthly negative.
                if i % 2 == 0:
                    prob_n[i] = fft.fftshift(np.abs(fft.irfft(prob_1_fft_n)))  # np.abs(fft.irfft(prob_1_fft_n))
                elif i % 2 == 1:
                    prob_n[i] = np.abs(fft.irfft(prob_1_fft_n))
                # fft.fftshift(np.abs(fft.irfft(prob_1_fft_n)))
                prob_1_fft_n *= self.dtemp  # Needed for normalization
        elif self.mode == 'log':
            prob_n = np.zeros((n + 1, len(prob_1)))
            convolution = prob_1
            for i in range(1, n + 1):
                prob_n[i] = convolution
                if i < n:
                    convolution = tools.convolve_log(convolution, prob_1, self.temp_range)
        else:
            print "Unknown mode: ", self.mode
            prob_n = np.zeros((n + 1, len(prob_1)))
        return prob_n

    # Probability that a voxel contains N sources
    def prob_of_n_sources(self, n, mean_n, sigma_g_squared):
        return \
            integrate.quad(
                lambda mu: 1.0 / mu * self.prob_log_normal(mu / mean_n, sigma_g_squared) * self.poisson(n, mu),
                0, 100 * mean_n, epsrel=1e-6)[0]

    @staticmethod
    def prob_log_normal(x, sigma_squared):
        return 1 / np.sqrt(2 * np.pi * sigma_squared) * np.exp(
            -1.0 / (2 * sigma_squared) * (np.log(x) + sigma_squared / 2) ** 2)

    # Poisson PMF
    @staticmethod
    def poisson(n, local_avg_n):
        return local_avg_n ** n * np.exp(-local_avg_n) / np.math.factorial(n)

    # # Poisson PMF
    # @staticmethod
    # def poisson2(n, local_avg_n):
    #     if n < 50:
    #         return local_avg_n ** n * np.exp(-local_avg_n) / np.math.factorial(n)
    #     elif n >= 50:
    #         return 1.0 / (np.sqrt(2 * np.pi) * local_avg_n) * np.exp(-(n - local_avg_n) ** 2 / (2 * local_avg_n))

    def default_luminosity_function(self, luminosity, fiducial_values, fiducial_units):
        fid_val = fiducial_units * fiducial_values
        return fid_val[0] * (luminosity / fid_val[1]) ** fid_val[2] * np.exp(
            -luminosity / fid_val[1] - fid_val[3] / luminosity)
