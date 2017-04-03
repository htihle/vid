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

infile = open('cubes/luminosity_function_vel.dat', 'rb')
infile.seek(0, 2)  # 2 corresponds to end of file
infile_size = infile.tell()
nhalos = (infile_size) / (4 * 7)  # 7 floats per halo
infile.seek(0)
print "nhalos = ", nhalos
lum_func = np.fromfile(infile, dtype=np.float32, count=nhalos * 7)
lum_func = np.reshape(lum_func, (nhalos, 7))

x = lum_func[:, 0]
y = lum_func[:, 1]
z = lum_func[:, 2]
vel = lum_func[:, 3]
redshift = lum_func[:, 4]
mass = lum_func[:, 5]
Lco = lum_func[:, 6]

n_vox = 128*128*1000
x, y, z = my_vid.get_grid(128)
volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])
# / (dL * n_vox * myVID.vol_vox)

bins_list = np.logspace(0, 8, 101)

phi_hist = np.histogram(Lco, bins=bins_list)[0]
lum_arr = (bins_list[1:] + bins_list[:-1]) / 2.0
dlum = bins_list[1:] - bins_list[:-1]

plt.loglog(lum_arr, lum_arr * phi_hist / (dlum * volume), label='CITA-luminosities')
plt.loglog(lum_arr, lum_arr * VID.VoxelIntensityDistribution.default_luminosity_function(
    lum_arr, best_fit_regular), label='best fit model to CITA-cube')
plt.loglog(lum_arr, lum_arr * VID.VoxelIntensityDistribution.default_luminosity_function(
    lum_arr, br_val), label='Breysse et al fit to Tony Li')
plt.loglog(lum_arr, lum_arr * mod_phi(lum_arr, best_fit_mod), label='best fit with extra parameters')
plt.legend(loc='lower left')
plt.title('Luminosity function comparison')
plt.xlabel(r'$L$ $[L_\odot]$')
plt.ylabel(r'$L\Phi$ [Mpc/h${}^{-3}$]')
plt.axis([1e2, 1e7, 1e-5, 1e-1])
plt.savefig('LumFuncCompare.pdf')
plt.show()