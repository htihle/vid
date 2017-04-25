import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

import tools
import VID

if len(sys.argv) > 1:
    filenames = sys.argv[1:]
else:
    print 'Include filenames of sample-files on command line.'

n_files = len(filenames)

print 'Opened', n_files, 'files.'

all_samples = []

for filename in filenames:
    samples = np.load(filename)
    all_samples.append(samples)

# This part assumes all the sample-files have the same number of samples
# and must be adjusted accordingly if this is not the case.
all_samples = np.array(all_samples)
shape = all_samples.shape
all_samples = all_samples.reshape((shape[0] * shape[1], shape[2]))

fixed_L_min = True

if fixed_L_min:
    L_min = 1e4
    all_samples[:, 3] = L_min

bin_edges, bin_centers, bin_spacings = tools.log_bins(1e4, 1e7, 100)

my_vid = VID.VoxelIntensityDistribution()

Lco = tools.load_cita_lums()

n_vox = 128*128*1000
x, y, z = my_vid.get_grid(128)
volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])

phi_hist = np.histogram(Lco, bins=bin_edges)[0]

plt.loglog(bin_centers, bin_centers * phi_hist / (bin_spacings * volume), label='CITA-luminosities')

all_samples = all_samples[:, :4].transpose()
percentiles = tools.lumfunc_confidence_interval(all_samples)
plt.fill_between(bin_centers, bin_centers * percentiles[0], bin_centers * percentiles[2],
                 facecolor='grey', alpha=0.3, label='95 % confidence interval')
plt.loglog(bin_centers, bin_centers * percentiles[1], 'k', label='Median value')
br_val = np.array([2.8e-10, 2.1e6, -1.87, 50e2, 1])
plt.loglog(bin_centers, bin_centers * VID.VoxelIntensityDistribution.default_luminosity_function(
    bin_centers, br_val), label='Breysse et al fit to Tony Li')
plt.legend(loc='lower left')
plt.title('Luminosity function comparison')
plt.xlabel(r'$L$ $[L_\odot]$')
plt.ylabel(r'$L\Phi$ [Mpc/h${}^{-3}$]')
plt.axis([1.1e4, 1e7, 1e-6, 1e-1])
plt.show()