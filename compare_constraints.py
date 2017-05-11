import numpy as np
import matplotlib.pyplot as plt
import sys

import tools
import VID

#s1 = np.load('samples_fixedcut_1e4_25_cubes_64kres.npy')
s2 = np.load('samples_fixedcut_5e3_19_cubes_64kres.npy')
s3 = np.load('samples_freecut_6_cubes_2mres.npy')
s4 = np.load('samples2017-04-28_14:51:18samp40000phi_freecut_cube0.npy')



bin_edges, bin_centers, bin_spacings = tools.log_bins(1e4, 1e7, 100)

my_vid = VID.VoxelIntensityDistribution()

Lco = tools.load_cita_lums()

n_vox = 128*128*1000
x, y, z = my_vid.get_grid(128)
volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])

phi_hist = np.histogram(Lco, bins=bin_edges)[0]

plt.loglog(bin_centers, bin_centers * phi_hist / (bin_spacings * volume), 'r', label='Actual CITA-luminosities')


def plot_constraints(samples, shade_color='grey', median_color='k', label='95 % confidence interval', alpha=0.3):
    percentiles = tools.lumfunc_confidence_interval(samples)
    plt.fill_between(bin_centers, bin_centers * percentiles[0], bin_centers * percentiles[2],
                     facecolor=shade_color, alpha=alpha, label=label)
    plt.loglog(bin_centers, bin_centers * percentiles[1], median_color)

plot_constraints(s3, label='From CITA - simulations', alpha=0.4)
#plot_constraints(s2, shade_color='green', median_color='g', label=r'$L_{min}$ = 5e3', alpha=0.2)
#plot_constraints(s1, shade_color='blue', median_color='b', label='fixed 1e4', alpha=0.4)
#plot_constraints(s4.transpose(), shade_color='blue', median_color='b', label=r'$L_{min}$ free, cube 0', alpha=0.5)


br_val = np.array([2.8e-10, 2.1e6, -1.87, 50e2, 1])
plt.loglog(bin_centers, bin_centers * VID.VoxelIntensityDistribution.default_luminosity_function(
    bin_centers, br_val), label='Breysse et al fit to Tony Li')
plt.legend(loc='lower left')
plt.title('Luminosity function comparison')
plt.xlabel(r'$L$ $[L_\odot]$')
plt.ylabel(r'$L\Phi$ [Mpc/h${}^{-3}$]')
plt.axis([3e4, 1e7, 1e-6, 1e-1])
plt.savefig('constraints.pdf')
plt.show()