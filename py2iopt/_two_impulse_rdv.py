"""
Functions made available to user to optimize two-impulse rendezvous trajectories.
"""

import pygmo as pg

from ._optim import deltav_udp

class TwoImpulseRDV:
    def __init__(self, algo="ipopt", mu=1):
        """Define algo and dynamics environments"""
        self.algo = algo
        self.mu = mu
        return

    def setProblem(self):
        """Define initial and final conditions"""
        return

    def solve(self):
        """Define UDP, UDA, population, and evolve()"""
        return

    def plot_trajectory_3d(self):
        """Plot 3D trajectory of the rendezvous"""
        return

    def plot_primer_vector_magnitude(self):
        """Plot the magnitude of the primer vector"""
        return

    def pretty_setting(self):
        """Display of the settings"""
        return

    def pretty_results(self):
        """Display of the outputs"""
        return 





    

    