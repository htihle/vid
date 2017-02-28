import numpy as np
import matplotlib.pyplot as plt
import ConfigParser

import emcee
import corner

import VID


def my_lnprob(par, dtemp_times_n_vox, vid, data):
    return -np.sum((vid.calculate_vid(par, x)[np.where(data > 0)] * dtemp_times_n_vox - data[
        np.where(data > 0)]) ** 2 / (2 * data[np.where(data > 0)]))


def do_mcmc_sampling(lnprob, vid, parameters, data, temp_range, n_vox, n_walkers, n_samples, n_burn_in=None, threads=1, autosave=True):
    from datetime import datetime
    now = str(datetime.now())
    time_string = now[:10] + '_' + now[11:19]
    comment = ''
    if n_burn_in is None:
        n_burn_in = n_samples / 5

    dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox

    ndim = len(parameters)
    p0 = np.array([parameters * (1 + 0.05 * np.random.randn(ndim)) for i in xrange(n_walkers)])

    sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, threads=threads,args=(dtemp_times_n_vox,vid,data))

    # burn-in.
    pos, prob, state = sampler.run_mcmc(p0, n_burn_in)
    sampler.reset()

    # main run
    sampler.run_mcmc(pos, n_samples, rstate0=state)

    values = np.zeros(ndim)
    for i in range(ndim):
        values[i] = corner.quantile(sampler.flatchain[:, i], [0.5])
    print values

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    if autosave:
        np.save('autosaves/values' + time_string, values)
        np.save('autosaves/samples' + time_string + 'samp' + str(n_samples * n_walkers) + comment, sampler.flatchain)
    fig = corner.corner(sampler.flatchain,
                        labels=[r"$\phi_* / 10^{-10}$", r"$L_* / 10^{6}$ ", r"$\alpha$", r"$L_{min} / 10^{2}$ ",
                                r"$\sigma_G$"], show_titles=True
                        , levels=(1 - np.exp(-0.5 * np.arange(1, 2.1, 1))))
    fig.savefig('ConstraintsFromVID' + comment + '.pdf')
    plt.show()


    plt.loglog(x, PofT / dtemp_times_n_vox, x, vid.calculate_vid(values, x))
    plt.show()

    print "Autocorrelation time:", sampler.get_autocorr_time(c=2)
    return 0


def full_phi(luminosity, fiducial_values, fiducial_units):
    fid_val = fiducial_units * fiducial_values
    return fid_val[0] * (luminosity / fid_val[1]) ** fid_val[2] * np.exp(
        -luminosity / fid_val[1] - fid_val[3] / luminosity)

config = ConfigParser.ConfigParser()
config.read('parameters.ini')

fid_val = [7.76, 1.34, -1.52, 5.17, 0.79]
fid_units = [1e-10, 1e6, 1, 1e2, 1]

myVID = VID.VoxelIntensityDistribution(full_phi, fid_val, fid_units, config)

n = 100

binsList = np.logspace(np.log10(5e-9), -4, n + 1)
box = np.load('tcube.npz')
flatBox = box.f.t.flatten()

dT = np.zeros(n)
x = np.zeros(n)
for i in range(n):
    dT[i] = binsList[i + 1] - binsList[i]
    x[i] = (binsList[i + 1] + binsList[i]) / 2.0

PofT = np.histogram((flatBox + 1e-12) * 1e-6, bins=binsList)[0]

do_mcmc_sampling(my_lnprob, myVID, np.array(fid_val), PofT, binsList, 400 * 25 * 25, 10, 5, threads=4)
