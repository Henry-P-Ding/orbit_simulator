from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from orbit_simulator.orbit_simulator import Simulation


class OrbitAnimation:
    def __init__(self, simulation: Simulation, fig: Figure, ax: Axes) -> None:
        self._sim = simulation
        self._fig = fig
        self._ax = ax
        self._animation = None

    def get_update_function(self):


    def animate(self):
        if self._animation is None:
            self._animation = FuncAnimation(self._fig, update, frames=total_frames)
