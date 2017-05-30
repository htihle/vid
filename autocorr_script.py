import numpy as np
import matplotlib.pyplot as plt
import VID
import tools
import MapMaker
import os.path

n = 100

temp_range = np.logspace(-8, -4, n + 1)
n_cubes = 25

tools.autocorr_from_cubes('cita_cube', temp_range, n_cubes)
#tools.autocorr_from_cubes('custom_sigma_1_alpha_neg1_cube', temp_range, n_cubes)
#tools.autocorr_from_cubes('custom_sigma_1_alpha_neg3_cube', temp_range, n_cubes)

my_vid = VID.VoxelIntensityDistribution()

x, y, z = my_vid.get_grid(128)
my_mapmaker = MapMaker.MapMaker(x, y, z)

a = [-1, -2] #, -3.0, -3.2]
c = [100.0] #, 40.0, 42.0]  #[10, 20, 50]
s = [1.0]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for a_val in a:  # when I'm less lazy, this could be paralellized very easily!
    for c_val in c:
        for s_val in s:
            sigma_g = s_val
            alpha_ps = a_val
            cut_length = c_val  # mpc/h
            cutoff = 1.0/cut_length

            cubename = 'custom_cube_s_' + str(sigma_g) + '_a_' + str(alpha_ps) + '_c_' + str(cut_length)
            if not os.path.isfile('cubes/' + cubename + '.npz'):
                ps_args = dict(alpha=alpha_ps, cutoff=cutoff)
                my_mapmaker.generate_cube(sigma_g=sigma_g, cubename=cubename, ps_args=ps_args, save_cube=True)

                tools.split_cube(cubename=cubename)

            label = r'$\alpha$ = ' + str(alpha_ps) + ', $\sigma_G$ = ' + str(sigma_g) \
                    + ', cut = ' + str(cut_length) + ' mpc'
            print label
            tools.autocorr_from_cubes(cubename, temp_range, n_cubes, label=label)

a = [-3] #, -3.0, -3.2]
c = [10.0, 40.0, 100.0] #, 40.0, 42.0]  #[10, 20, 50]
s = [1.0]  # [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for a_val in a:  # when I'm less lazy, this could be paralellized very easily!
    for c_val in c:
        for s_val in s:
            sigma_g = s_val
            alpha_ps = a_val
            cut_length = c_val  # mpc/h
            cutoff = 1.0/cut_length

            cubename = 'custom_cube_s_' + str(sigma_g) + '_a_' + str(alpha_ps) + '_c_' + str(cut_length)
            if not os.path.isfile('cubes/' + cubename + '.npz'):
                ps_args = dict(alpha=alpha_ps, cutoff=cutoff)
                my_mapmaker.generate_cube(sigma_g=sigma_g, cubename=cubename, ps_args=ps_args, save_cube=True)

                tools.split_cube(cubename=cubename)

            label = r'$\alpha$ = ' + str(alpha_ps) + ', $\sigma_G$ = ' + str(sigma_g) \
                    + ', cut = ' + str(cut_length) + ' mpc'
            print label
            tools.autocorr_from_cubes(cubename, temp_range, n_cubes, label=label)


plt.axhline(0, color='k', linestyle='--')
plt.xlabel(r'$\Delta T$ [dex]')
plt.ylabel(r'$A(\Delta T)$')
plt.legend()
plt.show()
