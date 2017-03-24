import numpy as np
import matplotlib.pyplot as plt
import sys
import corner
import ConfigParser

import VID


# Load autosaves
name = sys.argv[1]
cubename = sys.argv[2]
value_string = name[:10] + "values" + name[17:36] +".npy"

values = np.load(value_string)

print values

samples = np.load(name)


fig = corner.corner(samples, labels=[r"$\phi_* / 10^{-10}$",
                                     r"$L_* / 10^{6}$ ",
                                     r"$\alpha$",
                                     r"$L_{min} / 10^{2}$ ",
                                     r"$\sigma_G$"],
                    show_titles=True, levels=(1 - np.exp(-0.5 * np.arange(1, 2.1, 1))))
fig.savefig("CornerPlot.pdf")
plt.show()


def full_phi(luminosity, fiducial_values, fiducial_units):
    fid_val = fiducial_units * fiducial_values
    return fid_val[0] * (luminosity / fid_val[1]) ** fid_val[2] * np.exp(
        -luminosity / fid_val[1] - fid_val[3] / luminosity)

cube = np.load(cubename)

n_vox = len(cube.f.t.flatten())#25 * 25 * 100

n = 100

config = ConfigParser.ConfigParser()
config.read('parameters.ini')

fid_val = [7.76, 1.34, -1.52, 5.17, 0.79]
fid_units = [1e-10, 1e6, 1, 1e2, 1]
 

vid = VID.VoxelIntensityDistribution(full_phi, fid_val, fid_units, config)

temp_range = np.logspace(np.log10(2e-9), -4, n + 1)
B_i = np.histogram((cube.f.t.flatten() + 1e-12) * 1e-6, bins=temp_range)[0]
dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
x = (temp_range[1:] + temp_range[:-1]) / 2

plt.loglog(x, B_i / dtemp_times_n_vox, x, vid.calculate_vid(values, x))
plt.legend(['actual', 'best fit model'], loc='lower left')
plt.title('VID comparison Full CITA-cube')
plt.xlabel('T [K]')
plt.ylabel('P(T)')
plt.savefig('BestFitvsActual.pdf')
plt.show()
