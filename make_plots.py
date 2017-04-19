import numpy as np
import matplotlib.pyplot as plt
import sys
import corner
import ConfigParser

import VID
import tools


# Load autosaves
name = sys.argv[1]
cubename = sys.argv[2]
value_string = name[:10] + "values" + name[17:36] +".npy"

values = np.load(value_string)

print values

samples = np.load(name)

samples[:, 0] /= 1e-10
samples[:, 1] /= 1e6

# lab = [r"$\phi_* / 10^{-10}$",
#        r"$L_* / 10^{6}$ ",
#        r"$\alpha$",
#        r"$L_{min} / 10^{2}$ ",
#        r"$\sigma_G$"]

lab = [r"$\phi_* / 10^{-10}$",
       r"$L_* / 10^{6}$ ",
       r"$\alpha$",
       r"$\sigma_G$"]

fig = corner.corner(samples, labels=lab,
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

 

vid = VID.VoxelIntensityDistribution()

parm = np.zeros(5)
parm[0:3] = values[0:3]
parm[-1] = values[-1]
parm[3] = 100e2

vid_from_cube, x = tools.vid_from_cube('cubes/cita_cube_0.npz', add_noise=True, noise_temp=15.3, temp_range=np.logspace(np.log10(1e-5), -4, 100 + 1))

plt.loglog(x, vid_from_cube, x, vid.calculate_vid(parameters=parm, temp_array=x))
plt.legend(['actual', 'best fit model'], loc='lower left')
plt.title('VID comparison Full CITA-cube')
plt.xlabel('T [K]')
plt.ylabel('P(T)')
plt.savefig('BestFitvsActual.pdf')
plt.show()
