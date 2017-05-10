import numpy as np
import scipy.interpolate as interpolate
import os

import emcee
import corner

import VID
import tools

# my_vid = VID.VoxelIntensityDistribution()

# Lco = tools.load_cita_lums()
#
# bin_edges, bin_centers, bin_spacings = tools.log_bins(0.9e0, 1.1e8, 200)
#
# phi_hist = np.histogram(Lco, bins=bin_edges)[0]
#
# n_vox = 128*128*1000
# x, y, z = my_vid.get_grid(128)
# volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])
#
# lum_func = interpolate.interp1d(bin_centers, phi_hist / (bin_spacings * volume))

#
# def my_lnprob(par, dtemp_times_n_vox, vid, data):
#     if (par[-1] < 0) or (par[0] < 0) or (par[1] < 0) or (par[3] < 0):
#         return -np.inf
#     return -np.sum((vid.calculate_vid(parameters=par, temp_array=x)[np.where(data > 0)]
#                     - data[np.where(data > 0)]) ** 2 / (2 * data[np.where(data > 0)] / dtemp_times_n_vox))


# def my_lnprob(par, dtemp_times_n_vox, vid, data):
#     if (par[-1] < 0) or (par[0] < 0) or (par[1] < 0):
#         return -np.inf
#     parm = np.zeros(5)
#     parm[0:3] = par[0:3]
#     parm[-1] = par[-1]
#     parm[3] = 1e3
#     return -np.sum((vid.calculate_vid(parameters=parm, temp_array=x)[np.where(data > 0)]
#                     - data[np.where(data > 0)]) ** 2 / (2 * data[np.where(data > 0)] / dtemp_times_n_vox))


def do_mcmc_sampling(lnprob, parameters, lnprob_args, n_walkers, n_samples,
                     n_burn_in=None, threads=1, autosave=True, comment=''):
    from datetime import datetime
    now = str(datetime.now())
    time_string = now[:10] + '_' + now[11:19]
    if n_burn_in is None:
        n_burn_in = n_samples / 5

    ndim = len(parameters)
    p0 = np.array([parameters * (1 + 0.05 * np.random.randn(ndim)) for i in xrange(n_walkers)])

    sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, threads=threads, args=lnprob_args)

    # burn-in.
    pos, prob, state = sampler.run_mcmc(p0, n_burn_in)
    sampler.reset()

    # main run
    sampler.run_mcmc(pos, n_samples, rstate0=state)

    values = np.zeros(ndim)
    for i in range(ndim):
        values[i] = corner.quantile(sampler.flatchain[:, i], [0.5])
    print values

    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)
    if autosave:
        if not os.path.isdir('autosaves/'):
            os.makedirs('autosaves/')
        np.save('autosaves/values' + time_string, values)
        np.save('autosaves/samples' + time_string + 'samp' + str(n_samples * n_walkers) + comment, sampler.flatchain)
    try:
        print "Autocorrelation time:", sampler.get_autocorr_time(c=2)
    except emcee.autocorr.AutocorrError:
        print "Chain is to short to calculate autocorrelation time"
        pass
    return 0

# # par = [1.0]
# # best_fit_regular = [10.13348524e-10,   1.07747679e6,  -1.5833037,   13.50122216e2,   0.03816793]
# best_fit_fixedcut = [10.13348524e-10,   1.07747679e6,  -1.5833037,   0.03816793]
# n = 100
#
# binsList = np.logspace(np.log10(1e-5), -4, n + 1)
#
# for i in range(25):
#
#     data, x = tools.vid_from_cube('cubes/cita_cube_' + str(i) + '.npz', temp_range=binsList, add_noise=True, noise_temp=15.3)
#
#     do_mcmc_sampling(my_lnprob, my_vid, np.array(best_fit_fixedcut), data, binsList,
#                      1000 * 25 * 25, 50, 200, threads=4, message=str(i))
