from typing import Iterable

import numpy as np
import numpy.typing as npt
import scipy.constants
from scipy.integrate import solve_ivp


class Simulation:
    def __init__(self, masses: npt.ArrayLike, positions: npt.ArrayLike, velocities: npt.ArrayLike, time_range: tuple[float, float], time_points: Iterable[float], method='Rk23') -> None:
        self.masses = np.array(masses)
        self.positions = np.array(positions)
        self.velocities = np.array(velocities)
        self.time_range = time_range
        self.time_points = time_points
        self.method = 'RK23'

        if not self.masses.ndim == 1:
            raise ValueError(f"Number of dimensions for masses should be 1, but is {self.masses.ndim}")
        if not self.positions.ndim == 2:
            raise ValueError(f"Number of dimensions for positions should be 1, but is {self.positions.ndim}")
        if not self.velocities.ndim == 2:
            raise ValueError(f"Number of dimensions for velocities should be 1, but is {self.velocities.ndim}")
        if not all(map(lambda shape : self.masses.shape[0] == shape[0], [self.positions.shape, self.velocities.shape])):
            raise ValueError(f"shape[0] for masses, positions, and velocities should be match, but are {self.masses.shape[0]}, {self.positions.shape[0]}, {self.velocities.shape[0]}")

        self._position_solution = None

    @property
    def position_solution(self) -> npt.NDArray[np.float64]:
        if self._position_solution is None:
            self._position_solution = self.calculate_solution()
        return self._position_solution

    def get_initial_state(self) -> npt.NDArray[np.float64]:
        """
        Converts row-wise vector of velocities/positions into an initial state for the ODE solver

        The initial state is a single column-wise velocity-position vector with shape 6xN for N gravitational bodies.
        """
        return np.append(self.velocities, self.positions, axis=1).T

    def calculate_derivatives(self, t: float, initial_state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        vel_pos_mat = initial_state.reshape((6, self.masses.shape[0]), order='F')
        mass2_grid = np.tile(self.masses, (self.masses.shape[0], 1))

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
        mass2_grid = np.repeat(np.expand_dims(mass2_grid, 2), 3, axis=2)
        rel_cub_dist = np.repeat(np.expand_dims(rel_cub_dist, 2), 3, axis=2)

        grav_acc = -1 * scipy.constants.G * mass2_grid * rel_pos / rel_cub_dist
        grav_acc = np.nan_to_num(grav_acc)
        grav_acc_per_body = np.sum(grav_acc, axis=1)
        derivatives = np.append(grav_acc_per_body.T, vel_pos_mat[:3, :], axis=0)

        return derivatives.flatten(order='F')

    def calculate_solution(self) -> npt.NDArray[np.float64]:
        initial_state = self.get_initial_state()
        solution = solve_ivp(self.calculate_derivatives, self.time_range, initial_state.flatten(order='F'), t_eval=self.time_points, method=self.method)
        return solution.y.reshape(self.masses.shape[0], 6, solution.t.shape[0])[:, 3:, :]


"""
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
"""