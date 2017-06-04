import numpy as np
from mpi4py import MPI
import warnings

import VID
import tools
import MCMC


comm = MPI.COMM_WORLD

# print "Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size)
my_rank, size = (comm.Get_rank(), comm.Get_size())


def my_lnprob(par, n_vox, vid, data):
    if (par[-1] < 0) or (par[0] < 0) or (par[1] < 2e4) or (par[2] < -3.5):
        return - np.inf
    parm = np.zeros(5)
    parm[0:3] = par[0:3]
    parm[-1] = par[-1]
    parm[3] = 1e4
    B_i = vid.calculate_vid(parameters=parm, temp_array=bin_edges, subtract_mean_temp=True, bin_counts=True) * n_vox
    warnings.filterwarnings('error')
    try:
        return_value = -np.sum((B_i - data) ** 2 / (2 * B_i))
    except RuntimeWarning:
        print "RuntimeWarning caught, returning - np.inf"
        print "parm = ", parm
        return_value = - np.inf
    warnings.filterwarnings('default')
    return return_value

my_vid = VID.VoxelIntensityDistribution()

best_fit_fixedcut = [10.13348524e-10, 1.07747679e6, -1.5833037, 0.03816793]
n = 100

bin_edges, bin_centers, bin_spacings = tools.log_bins(1e-5, 1e-4, n)

n_cubes = 25

my_indices = tools.distribute_indices(n_cubes, size, my_rank)
print "Rank %d with indices" % my_rank, my_indices

n_vox = 1000 * 25 * 25

data, x = tools.vid_from_cube('cubes/cita_cube_' + str(2) + '.npz', temp_range=bin_edges, add_noise=True,
                              noise_temp=15.3, subtract_mean=True, bin_count=True)


for i in my_indices:
    data, x = tools.vid_from_cube('cubes/cita_cube_' + str(i) + '.npz', temp_range=bin_edges, add_noise=True,
                                  noise_temp=15.3, subtract_mean=True, bin_count=True)

    MCMC.do_mcmc_sampling(my_lnprob, np.array(best_fit_fixedcut), (n_vox, my_vid, data), 10, 20, threads=4,
                          comment='cube' + str(i))
