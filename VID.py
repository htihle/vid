import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
import numpy.fft as fft
import ast
import ConfigParser
import warnings

import tools


class VoxelIntensityDistribution:
    nu_em = 115e9  # 115 GHz
    speed_of_light = 3e8
    k_boltz = 1.3807e-23  # J/K
    mpc = 3.086e22  # 1 mpc = 3.086e22 m
    solar_lum = 3.848e26  # W

    def __init__(self, parameterfile=None, n_temp=None):
        config = ConfigParser.ConfigParser()
        if parameterfile is None:
            config.read('parameters.ini')
        else:
            config.read(parameterfile)

        self.config = config
        self.mode = config.get('Grid', 'mode')
        temp_max = float(config.get('Grid', 'temp_max'))
        if n_temp is None:
            self.n_temp = int(config.get('Grid', 'n_temp'))
        else:
            self.n_temp = n_temp
        if (self.mode == 'noise') or (self.mode == 's+n'):
            self.temp_range = np.linspace(-temp_max, temp_max, self.n_temp)
            self.dtemp = self.temp_range[1] - self.temp_range[0]
            self.sigma_noise = float(config.get('Telescope', 'sigma_noise')) * 1e-6
        elif self.mode == 'signal':
            self.temp_range = np.linspace(0, temp_max, self.n_temp)
            self.dtemp = self.temp_range[1] - self.temp_range[0]

        self.h = float(config.get('Cosmology', 'h'))
        self.omega_m = float(config.get('Cosmology', 'omega_m'))
        self.omega_l = float(config.get('Cosmology', 'omega_l'))
        self.n_max = int(config.get('Cosmology', 'n_max'))

        self.delta_nu = float(config.get('Telescope', 'delta_nu'))
        self.angular_res = float(config.get('Telescope', 'angular_res')) * (np.pi / (180 * 60))

        freq_max = float(config.get('Telescope', 'freq_max'))
        freq_min = float(config.get('Telescope', 'freq_min'))
        self.z_min = self.nu_em / freq_max - 1
        self.z_max = self.nu_em / freq_min - 1
        self.z_mid = (self.z_max + self.z_min) / 2.0
        self.vol_vox, self.x_lt = self.get_x_and_vol()
        self.n_freq = int(round((freq_max-freq_min)/self.delta_nu))

    # Function that calculates the relevant cosmology to translate between luminosity and brightness temperature.
    # Note that we do not take into account the changes in the vid at different redshift,
    # but we calculate the vid for the central redshift.
    def get_x_and_vol(self):
        comoving_distance = self.speed_of_light * integrate.quad(self.get_one_over_hubble, 0, self.z_mid)[0]
        vol_vox = 1.0 / (self.mpc ** 3) * (comoving_distance * self.angular_res) ** 2 * self.speed_of_light * \
                  self.get_one_over_hubble(self.z_mid) * (1 + self.z_mid) ** 2 * self.delta_nu / self.nu_em
        #print "Volume of 1 voxel: %g [(mpc/h)^3]" % vol_vox
        x_lt = self.h ** 2 * self.solar_lum * 1.0 / (self.mpc ** 3) * self.speed_of_light ** 3 \
               * (1 + self.z_mid) ** 2 / (8 * np.pi * self.k_boltz * self.nu_em ** 3) \
               * self.get_one_over_hubble(self.z_mid)
        return vol_vox, x_lt  # both given in (mpc/h)^3

    # Generates a grid that can be used by the mapmaker with the same properties as the vid.
    def get_grid(self, n_xy=25):
        dr_vox = self.speed_of_light * self.get_one_over_hubble(self.z_mid)\
                 * (1.0 + self.z_mid) ** 2 * self.delta_nu/self.nu_em * 1.0 / self.mpc
        z = np.linspace(0, self.n_freq * dr_vox, self.n_freq + 1)
        dx_vox = self.speed_of_light * integrate.quad(self.get_one_over_hubble, 0, self.z_mid)[0] \
                 * self.angular_res * 1.0 / self.mpc
        x = np.linspace(0, n_xy * dx_vox, n_xy + 1)
        y = x
        return x, y, z

    def get_one_over_hubble(self, z):
        hubble = 3.241e-18 * np.sqrt(self.omega_m * (1 + z) ** 3 + self.omega_l)  # in units of h/s
        return 1.0 / hubble

    # Function that actually calculates the vid. Can take an array of temperatures to
    # return the vid at those temperatures. Slightly complicated default behaviour with many different cases.

    def calculate_vid(self, lum_func=None, parameters=None, temp_array=None,
                      check_normalization=False, sigma_noise=None, bin_counts=False, subtract_mean_temp=False):
        if sigma_noise is None:
            if (self.mode == 'noise') or (self.mode == 's+n'):
                sigma_noise = self.sigma_noise
        else:
            sigma_noise *= 1e-6
        if self.mode == 'noise':
            prob_total = 1.0 / np.sqrt(2 * np.pi * sigma_noise ** 2) \
                         * np.exp(- self.temp_range ** 2 / (2.0 * sigma_noise ** 2))
        else:
            arguments = None
            if parameters is None:
                sigma_g = 1.0
                if lum_func is None:
                    arguments = np.array(ast.literal_eval(self.config.get("Luminosity", "arguments"))).astype(float)

            elif len(parameters[:-1]) == 0:
                sigma_g = parameters[-1]
                if lum_func is None:
                    arguments = np.array(ast.literal_eval(self.config.get("Luminosity", "arguments"))).astype(float)
            else:
                sigma_g = parameters[-1]
                arguments = parameters[:-1]

            if lum_func is None:
                lum_func = self.default_luminosity_function

            prob_1 = np.zeros(self.n_temp)

            if arguments is not None:
                number_density = self.integral(lum_func, 1e0, 1e8, args=arguments)
                prob_1[np.where(self.temp_range > 0)] = self.vol_vox / (number_density * self.x_lt) * lum_func(
                    self.temp_range[np.where(self.temp_range > 0)] * self.vol_vox / self.x_lt, arguments)
            else:
                number_density = self.integral(lum_func, 1e0, 1e8)
                prob_1[np.where(self.temp_range > 0)] = self.vol_vox / (number_density * self.x_lt) * lum_func(
                    self.temp_range[np.where(self.temp_range > 0)] * self.vol_vox / self.x_lt)
            if number_density < 0:
                print "Negative number density."
            # Could do this more efficient memory-wise (relevant for high numbers of convolutions) here.
            n_sources = int(round(min(self.n_max, 5 + number_density * self.vol_vox * 10 * 10 ** sigma_g)))
            prob_n = self.prob_of_temp_given_n(prob_1, n_sources)
            prob_signal = np.zeros(self.n_temp)

            for i in range(1, n_sources + 1):
                prob_signal += prob_n[i] * self.prob_of_n_sources(i, number_density * self.vol_vox, sigma_g ** 2)

            if self.mode == 's+n':
                prob_noise = 1.0 / np.sqrt(2 * np.pi * sigma_noise ** 2) \
                             * np.exp(- self.temp_range ** 2 / (2 * sigma_noise ** 2))
                prob_total = self.prob_of_n_sources(0, number_density * self.vol_vox, sigma_g ** 2) * prob_noise \
                             + fft.fftshift(np.abs(fft.irfft(fft.rfft(prob_signal) * fft.rfft(prob_noise))) * self.dtemp)
            elif self.mode == 'signal':
                prob_total = prob_signal

        if check_normalization:
            prob_func = interpolate.interp1d(self.temp_range, prob_total)
            norm = self.integral(prob_func, self.temp_range[0], self.temp_range[-1])
            print "Integral of P(T) is :", norm
            if self.mode == 'signal':
                prob_0 = self.prob_of_n_sources(0, number_density * self.vol_vox, sigma_g ** 2)
                print "Probability of empty voxel", prob_0
                print "Sum is:", norm + prob_0
        mean_temp = 0
        if subtract_mean_temp:
            mean_temp = self.dtemp * np.sum(self.temp_range * prob_total)
            if np.abs(mean_temp) > 1e-4:
                print "Large mean temp: ", mean_temp, "returning 0"
                return 0
        if temp_array is None:
            return prob_total, self.temp_range - mean_temp
        else:  # Interpolate in log-space ?
            p_func = interpolate.interp1d(self.temp_range - mean_temp,
                                          prob_total)  # interpolate.splrep(self.temp_range, prob_signal, s=0)
            if bin_counts:
                # If bin_count is true, then temp_array is interpreted as the bin edges, not the bin centers.
                # Also, returns bin count per sample voxel!
                #cum_int = integrate.cumtrapz(p_func(temp_array), temp_array, initial=0)
                bin_count = np.zeros(len(temp_array) - 1)
                for i in xrange(len(temp_array) - 1):
                    bin_count[i] = integrate.quad(p_func, temp_array[i], temp_array[i + 1], epsrel=1e-6)[0]
                return bin_count
            else:
                return p_func(temp_array)  # interpolate.splev(temp_array, p_func, der=0)

    # Probability distribution of expected temperature from N sources given P1
    # (= Probability distribution of expected temperature from 1 source)
    # Returns an array of values for N = 1 to n_max

    def prob_of_temp_given_n(self, prob_1, n):
        if self.mode == 's+n':
            prob_1_fft = fft.rfft(prob_1)
            prob_n = np.zeros((n + 1, len(prob_1)))
            prob_1_fft_n = 1
            for i in range(1, n + 1):
                prob_1_fft_n *= prob_1_fft
                # This ensures that the convolution is implemented correctly,
                # since otherwise it will mirror the origin.
                if i % 2 == 0:
                    prob_n[i] = fft.fftshift(np.abs(fft.irfft(prob_1_fft_n)))
                elif i % 2 == 1:
                    prob_n[i] = np.abs(fft.irfft(prob_1_fft_n))
                prob_1_fft_n *= self.dtemp  # Needed for normalization
        elif self.mode == 'signal':
            prob_1_fft = fft.rfft(prob_1)
            prob_n = np.zeros((n + 1, len(prob_1)))
            prob_1_fft_n = 1
            for i in range(1, n + 1):

                prob_1_fft_n *= prob_1_fft
                # Abs, because fft +ifft can make ~zero into sligthly negative.
                prob_n[i] = np.abs(fft.irfft(prob_1_fft_n))
                prob_1_fft_n *= self.dtemp  # Needed for normalization
        else:
            print "Unknown mode: ", self.mode
            prob_n = np.zeros((n + 1, len(prob_1)))
        return prob_n

    # Probability that a voxel contains N sources
    def prob_of_n_sources(self, n, mean_n, sigma_g_squared):
        # Full integral range works poorly for low sigma values.
        # 1e3 * mean_n should be tolerable for sigma_g < 3.
        if sigma_g_squared < 0.05 ** 2:
            return stats.poisson.pmf(n, mean_n)
        else:
            # Integration limits should somehow depend on n for efficiency,
            # but I have not found a simple way that works.
            return \
                self.integral(
                    lambda mu: 1.0 / mu * self.prob_log_normal(mu / mean_n, sigma_g_squared) * stats.poisson.pmf(n, mu),
                    0, 100 * mean_n * np.sqrt(sigma_g_squared), epsrel=1e-6)
            # return \
            #     integrate.quad(
            #         lambda mu: 1.0 / mu * self.prob_log_normal(mu / mean_n, sigma_g_squared) * stats.poisson.pmf(n, mu),
            #         0, 20 * 10 ** np.sqrt(sigma_g_squared) * mean_n, epsrel=1e-6)[0]
        # if sigma_g_squared < 0.3:
        #     return \
        #         integrate.quad(
        #             lambda mu: 1.0 / mu * self.prob_log_normal(mu / mean_n, sigma_g_squared) * stats.poisson.pmf(n, mu),
        #             0, 30 * mean_n, epsrel=1e-6)[0]
        # else:
        #     return \
        #         integrate.quad(
        #             lambda mu: 1.0 / mu * self.prob_log_normal(mu / mean_n, sigma_g_squared) * stats.poisson.pmf(n, mu),
        #             0, 1e3 * mean_n, epsrel=1e-6)[0]

    @staticmethod
    def prob_log_normal(x, sigma_squared):
        return 1.0 / np.sqrt(2 * np.pi * sigma_squared) * np.exp(
            -1.0 / (2 * sigma_squared) * (np.log(x) + sigma_squared / 2.0) ** 2)

    @staticmethod
    def integral(func, x_low, x_high, args=None, epsrel=1e-6):
        warnings.filterwarnings('error')
        if args is None:
            try:
                integral = integrate.quad(func, x_low, x_high, epsrel=epsrel)[0]
            except integrate.IntegrationWarning:
                print 'IntegrationWarning raised, using fixed quadrature instead.'
                integral = integrate.fixed_quad(func, x_low, x_high, n=4000)[0]
            warnings.filterwarnings('default')
            return integral
        else:
            try:
                integral = integrate.quad(func, x_low, x_high, args=args, epsrel=epsrel)[0]
            except integrate.IntegrationWarning:
                print 'IntegrationWarning raised, using fixed quadrature instead.'
                integral = integrate.fixed_quad(func, x_low, x_high, args=[args], n=4000)[0]
            warnings.filterwarnings('default')
            return integral
    # # Poisson PMF
    # @staticmethod
    # def poisson(n, local_avg_n):
    #     return local_avg_n ** n * np.exp(-local_avg_n) / np.math.factorial(n)

    # Fiducial model for the luminosity function from Breysse et. al.
    @staticmethod
    def default_luminosity_function(luminosity, args):
        return args[0] * (luminosity / args[1]) ** args[2] * np.exp(
            -luminosity / args[1] - args[3] / luminosity)
