import numpy as np
import matplotlib.pyplot as plt

import tools
import VID


def mod_phi(luminosity, fid_val):
    return fid_val[0] * ((luminosity / fid_val[1]) ** fid_val[3] + (luminosity / fid_val[2]) ** fid_val[4]) * np.exp(
        -luminosity / fid_val[1] - fid_val[5] / luminosity)

best_fit_mod = [43.37e-10, 0.4085648e6, 2.96995518e5, -0.23084209, -1.6637809, 19.40474563e2, 0.82880997]
best_fit_regular = [10.13348524e-10,   1.07747679e6,  -1.5833037,   13.50122216e2,   0.03816793]#[7.76e-10,   1.34e+06,  -1.52e+00,   5.17e+02, 0.79]
br_val = np.array([2.8e-10, 2.1e6, -1.87, 50e2, 1])

my_vid = VID.VoxelIntensityDistribution()

vid, x = tools.vid_from_cube('cubes/cita_cube.npz')

plt.loglog(x, vid, label='cita-cube vid')
plt.loglog(x, my_vid.calculate_vid(parameters=best_fit_regular, temp_array=x), label='best fit regular lumfunc')
plt.loglog(x, my_vid.calculate_vid(lum_func=mod_phi, parameters=best_fit_mod, temp_array=x),
           label='best fit with extra parameters')
plt.legend(loc='lower left')
plt.title('Vid comparison')
plt.xlabel(r'$T$ [K]')
plt.ylabel(r'$P(T)$ [K${}^{-1}$]')
#plt.axis([1e2, 1e7, 1e-5, 1e-1])
plt.savefig('VidCompare.pdf')
plt.show()