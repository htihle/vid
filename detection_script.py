import numpy as np
import scipy.stats as stats

import tools


def calculate_detection_significance(cube_name, sigma_noise, temp_range, n_iters):
    n_samp = len(temp_range) - 1
    chi2 = np.zeros(n_iters)
    chi2_nofit = np.zeros(n_iters)
    for i in range(n_iters):
        noise_cube = np.random.randn(25, 25, 1000) * sigma_noise
        n_vox = len(noise_cube.flatten())
        dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
        #vid, x = tools.vid_from_cube(cube=noise_cube, temp_range=temp_range)
        vid, x = tools.vid_from_cube(cube_name, add_noise=True, noise_temp=sigma_noise,
                                     temp_range=temp_range, subtract_mean=True)
        best_sigma = tools.best_fit_noise(vid, x, sigma_noise, dtemp_times_n_vox)
        model = tools.noise_vid(x, sigma_noise=best_sigma*1e-6)
        model_nofit = tools.noise_vid(x, sigma_noise=sigma_noise*1e-6)
        chi2[i] = tools.calculate_chi_squared(vid, model, model / dtemp_times_n_vox)
        chi2_nofit[i] = tools.calculate_chi_squared(vid, model_nofit, model_nofit / dtemp_times_n_vox)

    val = np.mean(chi2)
    val_nofit = np.mean(chi2_nofit)
    sf = stats.chi2.sf(val, n_samp)
    log_sf = stats.chi2.logsf(val, n_samp)
    log_sf_nofit = stats.chi2.logsf(val_nofit, n_samp)
    print val
    print (val - stats.chi2.mean(n_samp))/stats.chi2.std(n_samp)
    print sf
    pte = np.exp(log_sf)
    print pte
    # significance = stats.norm.ppf(np.exp(log_sf))
    # print significance


    print "\nnofit \n"

    print val_nofit
    print (val_nofit - stats.chi2.mean(n_samp))/stats.chi2.std(n_samp)
    pte_nofit = np.exp(log_sf_nofit)
    print pte_nofit
    # significance_nofit = stats.norm.ppf(np.exp(log_sf_nofit))
    # print significance_nofit
    print '\n'
    return chi2, chi2_nofit  # val, pte, val_nofit, pte_nofit

sigma_noise_arr = [15.0, 20.0, 25.0, 30.0, 35.0]
n_samp = 50
temp_range = np.logspace(-5, -3.8, n_samp + 1)
n_iters = 100
results = np.zeros((len(sigma_noise_arr), 25, 2, n_iters))
for j in range(len(sigma_noise_arr)):
    sigma_noise = sigma_noise_arr[j]
    for i in range(25):
        print '\n'
        print 'j = ', j
        print 'i = ', i
        print '\n'
        results[j, i] = calculate_detection_significance('cubes/cita_cube_' + str(i) + '.npz',
                                                         sigma_noise, temp_range, n_iters)

np.save('significance_results_subtractmean_100', results)