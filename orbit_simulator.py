import timeit

import numpy as np
import scipy.constants
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def calculate_derivatives(t, vel_pos_vec):
    vel_pos_mat = vel_pos_vec.reshape(6, masses.shape[0], order='F')
    mass1_grid, mass2_grid = np.meshgrid(masses, masses)

    p1_grid_x, p2_grid_x = np.meshgrid(vel_pos_mat[3, :], vel_pos_mat[3, :])
    p1_grid_y, p2_grid_y = np.meshgrid(vel_pos_mat[4, :], vel_pos_mat[4, :])
    p1_grid_z, p2_grid_z = np.meshgrid(vel_pos_mat[5, :], vel_pos_mat[5, :])
    rel_pos = np.transpose(np.array([
        p2_grid_x - p1_grid_x,
        p2_grid_y - p1_grid_y,
        p2_grid_z - p1_grid_z
    ]), (1, 2, 0))
    rel_cub_dist = np.sum(rel_pos ** 2, axis=2) ** (3 / 2)
    # add additional 3-component dimension to relative distance and mass grids for force components
    mass1_grid = np.repeat(np.expand_dims(mass1_grid, 2), 3, axis=2)
    mass2_grid = np.repeat(np.expand_dims(mass2_grid, 2), 3, axis=2)
    rel_cub_dist = np.repeat(np.expand_dims(rel_cub_dist, 2), 3, axis=2)

    gravities = -1 * scipy.constants.G * mass1_grid * mass2_grid * rel_pos / rel_cub_dist
    gravities = np.nan_to_num(gravities)
    gravities_per_body = np.sum(gravities, axis=1)  # TODO: check if this results in the right direction
    acc_per_body = gravities_per_body / np.tile(masses, (3, 1)).T
    derivatives = np.append(acc_per_body.T, vel_pos_mat[:3, :], axis=0)

    return derivatives.flatten(order='F')


time_range = [0, 1e8]
time_step = 1e5
time_points = np.linspace(*time_range, int((time_range[1] - time_range[0]) / time_step))
print(len(time_points))
masses = np.array([5.683e26, 8.66e25, 1.898e27, 1.988e30, 1.027e26, 1.309e22, 6.39e23, 4.867e24, 3.285e23, 5.972e24])
initial_positions = np.array([[1.357e12, 0, -1.303e11], [2.732e12, 0, -3.083e11], [7.41e11, 0, -7.961e10], [0, 0, 0], [4.471e12, 0, -5.007e11],
     [4.434e12, 0, 9.128e11], [2.067e11, 0, -2.035e10], [1.075e11, 0, -7.237e9], [4.6e10, 0, -2.712e9],
     [1.471e11, 0, -1.832e10]])
initial_velocities = np.array([[0, 1.014e4, 0], [0, 7.13e3, 0], [0, 1.372e4, 0], [0, -16.79, 0], [0, 5.47e3, 0], [0, 6.1e3, 0], [0, 2.65e4, 0],
     [0, 3.526e4, 0], [0, 5.897e4, 0], [0, 3.029e4, 0]])
initial_state = np.append(initial_velocities, initial_positions, axis=1).T

trial_count = int(1e2)
benchmark_time = timeit.timeit(
    "solve_ivp(calculate_derivatives, time_range, initial_state.flatten(order='F'), t_eval=time_points, method='RK23')",
    globals=locals(),
    number=trial_count
) / trial_count
print(f"Average execution time over {trial_count} trial(s): {benchmark_time:e} s")
exit()

solution = solve_ivp(calculate_derivatives, time_range, initial_state.flatten(order='F'), t_eval=time_points, method='RK23')
print(solution.message)
t_points = solution.t
pos_solution = solution.y.reshape(masses.shape[0], 6, t_points.shape[0])[:, 3:, :]

fig, ax = plt.gcf(), plt.gca()
fig.set_dpi(100)
ax.set_xlim(np.min(pos_solution[:, 0, :]), np.max(pos_solution[:, 0, :]))
ax.set_ylim(np.min(pos_solution[:, 1, :]), np.max(pos_solution[:, 1, :]))
ax.set_aspect('equal')
ax.plot(0, 0, 'bo')
lines = [ax.plot([], [], '-')[0] for i in range(pos_solution.shape[0])]
dots = [ax.plot([], [], 'o')[0] for i in range(pos_solution.shape[0])]

time_point_index_frame_ratio = 1e3
total_frames = int(len(time_points) / time_point_index_frame_ratio)
def update(frame):
    print(f"Completion: {frame / total_frames * 100:.2f}")
    time_point = int(frame * time_point_index_frame_ratio)
    for i, line in enumerate(lines):
        line.set_data(pos_solution[i, 0, :time_point], pos_solution[i, 1, :time_point])
    for i, dot in enumerate(dots):
        dot.set_data(pos_solution[i, 0, time_point], pos_solution[i, 1, time_point])
    return lines

ani = FuncAnimation(fig, update, frames=total_frames)
ani.save('./animation.gif', writer='imagemagick', fps=30)
