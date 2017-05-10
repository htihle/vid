import numpy as np
from mpi4py import MPI

import VID
import tools
import MCMC


comm = MPI.COMM_WORLD

# print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)
my_rank, size = (comm.rank, comm.size)


def my_lnprob(par, sigma_squared, vid, data):
    if (par[-1] < 0) or (par[0] < 0) or (par[1] < 0):
        return -np.inf
    parm = np.zeros(5)
    parm[0:3] = par[0:3]
    parm[-1] = par[-1]
    parm[3] = 1e4
    local_vid = vid.calculate_vid(parameters=parm, temp_array=x)
    return -np.sum((local_vid[np.where(sigma_squared > 0)] - data[np.where(sigma_squared > 0)]) ** 2 /
                   (2 * sigma_squared[np.where(sigma_squared > 0)]))

my_vid = VID.VoxelIntensityDistribution()

best_fit_fixedcut = [10.13348524e-10, 1.07747679e6, -1.5833037, 0.03816793]
n = 100

bin_edges, bin_centers, bin_spacings = tools.log_bins(1e-5, 1e-4, n)

n_cubes = 25

my_indices = tools.distribute_indices(n_cubes, size, my_rank)
print "Rank %d with indices" % my_rank, my_indices

cita_vid = tools.vid_from_cube('cubes/cita_cube.npz', temp_range=bin_edges, add_noise=True,
                               noise_temp=15.3)[0]
n_vox = 1000 * 25 * 25

sigma_squared = cita_vid / (n_vox * bin_spacings)

for i in my_indices:
    data, x = tools.vid_from_cube('cubes/cita_cube_' + str(i) + '.npz', temp_range=bin_edges, add_noise=True,
                                  noise_temp=15.3)

    MCMC.do_mcmc_sampling(my_lnprob, np.array(best_fit_fixedcut), (sigma_squared, my_vid, data), 10, 20, threads=1,
                          comment='cube' + str(i))