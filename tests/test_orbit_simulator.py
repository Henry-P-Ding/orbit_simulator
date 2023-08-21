import unittest
import timeit

import numpy as np
from orbit_simulator.orbit_simulator import Simulation

# @formatter:off
# gravitational body masses (kg)
EXAMPLE_MASSES = np.array([
    5.683e26,
    8.660e25,
    1.898e27,
    1.988e30,
    1.027e26,
    1.309e22,
    6.390e23,
    4.867e24,
    3.285e23,
    5.972e24
])
# initial positions of gravitational bodies in Cartesian basis (m)
EXAMPLE_INITIAL_POSITIONS = np.array([
    [ 1.357e12,  0.000e01, -1.303e11],
    [ 2.732e12,  0.000e01, -3.083e11],
    [ 7.410e11,  0.000e01, -7.961e10],
    [ 0.000e01,  0.000e01,  0.000e01],
    [ 4.471e12,  0.000e01, -5.007e11],
    [ 4.434e12,  0.000e01,  9.128e11],
    [ 2.067e11,  0.000e01, -2.035e10],
    [ 1.075e11,  0.000e01, -7.237e09],
    [ 4.600e10,  0.000e01, -2.712e09],
    [ 1.471e11,  0.000e01, -1.832e10]
])
# initial velocities of gravitational bodies in Cartesian basis (m/s)
EXAMPLE_INITIAL_VELOCITIES = np.array([
    [ 0.000e01,  1.014e4,  0.000e01],
    [ 0.000e01,  7.130e3,  0.000e01],
    [ 0.000e01,  1.372e4,  0.000e01],
    [ 0.000e01, -1.679e1,  0.000e01],
    [ 0.000e01,  5.470e3,  0.000e01],
    [ 0.000e01,  6.100e3,  0.000e01],
    [ 0.000e01,  2.650e4,  0.000e01],
    [ 0.000e01,  3.526e4,  0.000e01],
    [ 0.000e01,  5.897e4,  0.000e01],
    [ 0.000e01,  3.029e4,  0.000e01]
])

EXAMPLE_INITIAL_DERIVATIVES = np.array([
    -7.13968889e-05,  0.00000000e+00,  6.85092053e-06,  0.00000000e+00,
     1.01400000e+04,  0.00000000e+00, -1.74916327e-05,  0.00000000e+00,
     1.97428424e-06,  0.00000000e+00,  7.13000000e+03,  0.00000000e+00,
    -2.37426329e-04,  0.00000000e+00,  2.55105872e-05,  0.00000000e+00,
     1.37200000e+04,  0.00000000e+00,  3.05395680e-07,  0.00000000e+00,
    -3.12635336e-08,  0.00000000e+00, -1.67900000e+01,  0.00000000e+00,
    -6.52936894e-06,  0.00000000e+00,  7.31246024e-07,  0.00000000e+00,
     5.47000000e+03,  0.00000000e+00, -6.35428487e-06,  0.00000000e+00,
    -1.31308966e-06,  0.00000000e+00,  6.10000000e+03,  0.00000000e+00,
    -3.06063717e-03,  0.00000000e+00,  3.01313945e-04,  0.00000000e+00,
     2.65000000e+04,  0.00000000e+00, -1.14034985e-02,  0.00000000e+00,
     7.67630312e-04,  0.00000000e+00,  3.52600000e+04,  0.00000000e+00,
    -6.23796911e-02,  0.00000000e+00,  3.67767049e-03,  0.00000000e+00,
     5.89700000e+04,  0.00000000e+00, -5.99177177e-03,  0.00000000e+00,
     7.46260277e-04,  0.00000000e+00,  3.02900000e+04,  0.00000000e+00
])
# @formatter:on


class TestOrbitSimulator(unittest.TestCase):
    def test_calculate_derivative(self):
        sim = Simulation(EXAMPLE_MASSES, EXAMPLE_INITIAL_POSITIONS, EXAMPLE_INITIAL_VELOCITIES, (0, 1),
                                         [0])
        initial_derivatives = sim.calculate_derivatives(0, sim.get_initial_state())
        for derivative_index, derivative in enumerate(initial_derivatives):
            self.assertAlmostEqual(derivative, EXAMPLE_INITIAL_DERIVATIVES[derivative_index], msg=f"Derivative at index {derivative_index} is wrong")

    def test_benchmark_solution(self):
        time_range = (0, 1e8)
        time_step = 1e5
        time_points = np.linspace(*time_range, int((time_range[1] - time_range[0]) / time_step))
        sim = Simulation(EXAMPLE_MASSES, EXAMPLE_INITIAL_POSITIONS, EXAMPLE_INITIAL_VELOCITIES, time_range, time_points)
        trial_count = int(1e2)
        benchmark_time = timeit.timeit(
            "sim.calculate_solution()",
            globals=locals(),
            number=trial_count
        ) / trial_count
        print(f"Average execution time over {trial_count} trial(s): {benchmark_time:e} s")


if __name__ == '__main__':
    unittest.main()
