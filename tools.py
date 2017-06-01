import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interpolate

import VID


# Calculates the angular average of any map.
def angular_average_3d(inmap, x, y, z, dr, x0=0, y0=0, z0=0):
    x_ind, y_ind, z_ind = np.indices(inmap.shape)

    r = np.sqrt((x[x_ind] - x0) ** 2
                + (y[y_ind] - y0) ** 2
                + (z[z_ind] - z0) ** 2)

    # np.hypot(x[x_ind] - x0, y[y_ind] - y0, z[z_ind] - z0)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] / dr
    map_sorted = inmap.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    delta_r = r_int[1:] - r_int[:-1]  # Assumes all dr intervals represented

    rind = np.where(delta_r)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(map_sorted, dtype=float)
    sum_rbin = csim[rind[1:]] - csim[rind[:-1]]

    return sum_rbin / nr, (r_int[rind[1:]] + 0.5) * dr  # average value of function in each radial bin of length dr


def generate_poisson_map(lambda_map):
    output_map = np.zeros_like(lambda_map)
    output_map[np.where(lambda_map < 0)] = 0
    output_map[np.where(lambda_map >= 0)] = np.random.poisson(lambda_map[np.where(lambda_map >= 0)])
    return output_map


# Inverse cumulative distribution function, used for inversion sampling of any distribution.
def get_inv_cdf(func, edges, log=True, args=None, return_norm=False):
    if log:
        n_cmf = 5000
        x = np.logspace(np.log10(edges[0]), np.log10(edges[1]), n_cmf)
    else:
        n_cmf = 10000
        x = np.linspace(edges[0], edges[1], n_cmf)
    if args is None:
        f_x = func(x)
    else:
        f_x = func(x, args)

    cdf = integrate.cumtrapz(f_x, x, initial=0)

    norm = cdf[-1]
    if norm < 0.99 or norm > 1.01:
        print "Pdf not exactly normalized on this interval, renormalizing. Norm = ", norm

    cdf /= norm

    inv_cdf_func = interpolate.interp1d(cdf, x)

    if return_norm:
        return inv_cdf_func, norm
    else:
        return inv_cdf_func


def vid_from_cube(cube_name=None, cube=None, temp_range=None, add_noise=False, noise_temp=0, subtract_mean=False,
                  bin_count=False):
    if temp_range is None:
        n = 100
        temp_range = np.logspace(-9, -4, n + 1)
    if cube is None:
        cube = (np.load(cube_name))
        cube = cube.f.t
    if add_noise:
        flat_cube = (cube + np.random.randn(*cube.shape) * noise_temp).flatten()  # cube.f.t.shape
    else:
        flat_cube = cube.flatten()
    if subtract_mean:
        flat_cube = flat_cube - np.mean(flat_cube)
    my_hist = np.histogram((flat_cube + 1e-12) * 1e-6, bins=temp_range)[0]
    n_vox = len(cube.flatten())
    dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
    x = (temp_range[1:] + temp_range[:-1]) / 2
    if bin_count:
        return my_hist, x
    else:
        return my_hist / dtemp_times_n_vox, x


def power_law_ps(k, alpha=0, cutoff=0):  # Always cuts zero-frequency.
    k_not_cut = k[np.where(k > cutoff)]
    out_ps = np.zeros_like(k)
    out_ps[np.where(k > cutoff)] = k_not_cut ** alpha
    return out_ps


# Calculates the autocorrelation between the different bins of the vid from a cube.
def autocorr_from_cubes(cubename, temp_range=None, n_cubes=25, label=None):
    if temp_range is None:
        temp_range = np.logspace(-8, -4, 101)

    n = len(temp_range) - 1
    cube = np.load("cubes/" + cubename + ".npz")
    myhist = np.histogram((cube.f.t.flatten() + 1e-12) * 1e-6, bins=temp_range)[0]
    n_vox_tot = len(cube.f.t.flatten())
    #dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
    x = (temp_range[1:] + temp_range[:-1]) / 2
    dex = np.log10(x[1]) - np.log10(x[0])
    avg = myhist

    B_i = []
    for i in range(n_cubes):
        cube = np.load("cubes/" + cubename + "_" + str(i) + ".npz")
        B = np.histogram((cube.f.t.flatten() + 1e-12) * 1e-6, bins=temp_range)[0]
        n_vox = len(cube.f.t.flatten())
        # dtemp_times_n_vox = (temp_range[1:] - temp_range[:-1]) * n_vox
        # x = (temp_range[1:] + temp_range[:-1]) / 2
        B_i.append(B)
    B_i = np.array(B_i)

    sigma = B_i.std(0)
    n_dt = int(0.9*n)

    autocorr = np.zeros(n_dt)

    for i in range(n_cubes):
        residual = (B_i[i] - avg * float(n_vox) / n_vox_tot)/sigma
        autocorr += np.correlate(residual, residual, 'full')[len(residual):len(residual) + n_dt]

    autocorr /= n_cubes * n

    dex_arr = np.linspace(0, n_dt * dex, n_dt)
    if label is None:
        plt.plot(dex_arr, autocorr, label=cubename + ' (avg)')
    else:
        plt.plot(dex_arr, autocorr, label=label)


# Split one large cube into alot of smaller ones. Assumes symmetry in x-y plane.
def split_cube(cubename, rows=5, nr=25):
    data_cube = np.load("cubes/" + cubename + ".npz")
    full_cube = data_cube.f.t
    for i in range(rows):
        for j in range(rows):
            index = rows * i + j
            # print "\n\ncubes/" + cubename + "_" + str(index)
            # print j*25, (j+1)*25, i*25, (i+1)*25
            np.savez("cubes/" + cubename + "_" + str(index),
                     t=full_cube[j * nr:(j + 1) * nr, i * nr:(i + 1) * nr, :])
    print "Cube is split!"


# Fit noise vid to any vid.
def best_fit_noise(vid, temp_range, sigma_noise, dtemp_times_n_vox):
    best_sigma = sigma_noise
    my_min = 1e13
    sigma_arr = np.linspace(sigma_noise * 0.60, sigma_noise * 1.4, 1000)
    for sigma in sigma_arr:
        model = noise_vid(temp_range, sigma_noise=sigma * 1e-6)
        chi2 = calculate_chi_squared(vid, model, std_2=model/dtemp_times_n_vox)
        if chi2 < my_min:
            my_min = chi2
            best_sigma = sigma
    return best_sigma


def calculate_chi_squared(data, model, std_2=1.0):
    return np.sum((data - model) ** 2 / std_2)


def noise_vid(temp, sigma_noise=1.0):
    return 1.0 / np.sqrt(2 * np.pi * sigma_noise ** 2) * np.exp(- temp ** 2 / (2 * sigma_noise ** 2))


def load_cita_lums(name='cubes/luminosity_function_vel.dat'):
    infile = open(name, 'rb')
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
    return Lco


def log_bins(x_start, x_end, n_bins):
    bin_edges = np.logspace(np.log10(x_start), np.log10(x_end), n_bins + 1)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    bin_spacings = bin_edges[1:] - bin_edges[:-1]

    return bin_edges, bin_centers, bin_spacings


def lumfunc_confidence_interval(samples, lum_range=None, percentiles=None):
    if lum_range is None:
        lum_range = np.logspace(4, 7, 100)
    if percentiles is None:
        percentiles = [2.5, 50, 97.5]

    return np.percentile(VID.VoxelIntensityDistribution.default_luminosity_function(lum_range[:, None], samples),
                         percentiles, axis=1)


def distribute_indices(n_indices, n_processes, my_rank):
    divide = n_indices / n_processes
    leftovers = n_indices % n_processes

    if my_rank < leftovers:
        my_n_cubes = divide + 1
        my_offset = my_rank
    else:
        my_n_cubes = divide
        my_offset = leftovers
    start_index = my_rank * divide + my_offset
    my_indices = range(start_index, start_index + my_n_cubes)
    return my_indices


def sample_from_vid(parameters, edges=None):
    if edges is None:
        edges = [1e-9, 5e-4]

    my_vid = VID.VoxelIntensityDistribution()

    vid = my_vid.calculate_vid(parameters=parameters, check_normalization=True)
    vid_func = interpolate.interp1d(my_vid.temp_range, vid)
    return get_inv_cdf(vid_func, edges=edges, return_norm=True)