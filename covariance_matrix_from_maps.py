import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import VID
import tools
import MapMaker

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n_cosmologies = 200

my_n_cosmologies = n_cosmologies / size
if n_cosmologies % size > 0:
    my_n_cosmologies += 1

# print "my_n", my_n_cosmologies

n_bins = 10
n_samples = 100000
alpha_ps = -3
cutoff = 1.0/40
fiducial_values = [1.0e-9,   1.0e6,  -1.6,   1.0e4,   1.0]

my_vid = VID.VoxelIntensityDistribution()

bin_edges, bin_centers, bin_spacings = tools.log_bins(1e-5, 1e-4, n_bins)

cube_hist = np.zeros((my_n_cosmologies, n_bins), dtype='i')
indep_hist = np.zeros((my_n_cosmologies, n_bins), dtype='i')
recvbuf = None

model_hist = my_vid.calculate_vid(parameters=fiducial_values, temp_array=bin_centers) * bin_spacings * n_samples

inv_cdf, norm = tools.sample_from_vid(fiducial_values)
samples_with_sources = int(round(n_samples * norm))
x, y, z = my_vid.get_grid(10)
ps_args = dict(alpha=alpha_ps, cutoff=cutoff)
my_mapmaker = MapMaker.MapMaker(x, y, z)
for i in range(my_n_cosmologies):
    cube = my_mapmaker.generate_cube(sigma_g=fiducial_values[-1], lum_args=fiducial_values,
                                     ps_args=ps_args, save_cube=False)[0] * 1e-6

    cube_hist[i, :] = np.histogram(cube.flatten(), bins=bin_edges)[0] - model_hist
    indep_hist[i, :] = np.histogram(inv_cdf(np.random.rand(samples_with_sources)), bins=bin_edges)[0] - model_hist
if rank == 0:
    recvbuf = np.empty([size * my_n_cosmologies, n_bins], dtype='i')
comm.Gather(cube_hist, recvbuf, root=0)
if rank == 0:
    print recvbuf.shape
    cov = np.cov(recvbuf, rowvar=False)
    print cov

if rank == 0:
    recvbuf = np.empty([size * my_n_cosmologies, n_bins], dtype='i')
comm.Gather(indep_hist, recvbuf, root=0)
if rank == 0:
    print recvbuf.shape
    cov_indep = np.cov(recvbuf, rowvar=False)
    print cov_indep

if rank == 0:
    plt.figure()
    plt.imshow(cov, interpolation='none')
    plt.colorbar()

    plt.figure()
    plt.imshow(cov_indep, interpolation='none')
    plt.colorbar()

    plt.show()