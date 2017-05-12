import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

import VID
import tools
import MapMaker

n_bins = 50
n_samples = 625000
alpha_ps = -3
cutoff = 1.0/40
fiducial_values = [1.0e-9,   1.0e6,  -1.6,   1.0e4,   2.0]

my_vid = VID.VoxelIntensityDistribution()

bin_edges, bin_centers, bin_spacings = tools.log_bins(1e-5, 1e-4, n_bins)


def sample_from_vid(parameters, n_samples, edges=None):
    if edges is None:
        edges = [1e-9, 5e-4]

    my_vid = VID.VoxelIntensityDistribution()

    vid = my_vid.calculate_vid(parameters=parameters, check_normalization=True)
    vid_func = interpolate.interp1d(my_vid.temp_range, vid)
    inv_cdf, norm = tools.get_inv_cdf(vid_func, edges=edges, return_norm=True)
    samples_with_sources = int(round(n_samples * norm))
    return inv_cdf(np.random.rand(samples_with_sources)), norm, samples_with_sources

s, norm, samples_with_sources = sample_from_vid(fiducial_values, n_samples)

random_hist = np.histogram(s, bins=bin_edges)[0]
model_hist = my_vid.calculate_vid(parameters=fiducial_values, temp_array=bin_centers) * bin_spacings * n_samples



plt.errorbar(x=bin_centers, y=model_hist, yerr=np.sqrt(model_hist))
plt.loglog(bin_centers, random_hist)

x, y, z = my_vid.get_grid(25)
ps_args = dict(alpha=alpha_ps, cutoff=cutoff)
my_mapmaker = MapMaker.MapMaker(x, y, z)
for i in range(6):
    cube = my_mapmaker.generate_cube(sigma_g=fiducial_values[-1], lum_args=fiducial_values,
                                     ps_args=ps_args, save_cube=False)[0] * 1e-6

    cube_hist = np.histogram(cube.flatten(), bins=bin_edges)[0]

    plt.loglog(bin_centers, cube_hist)
plt.show()