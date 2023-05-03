"""
Functions made available to user to optimize two-impulse rendezvous trajectories.
"""

import pygmo as pg

from ._optim import deltav_udp

class TwoImpulseRDV:
    def __init__(self, mu=1, algo="ipopt"):
        """Define dynamics environment and algorithm.
        
        Args:
            mu (float): central body's gravitational parameter
            algo (str): optimization algorithm
        """
        self.mu = mu
        self.algo = algo
        return

    def setProblem(self, t0, tf, r0, rf, v0, vf):
        """Define initial and final conditions of the rendezvous.
        
        Args:
            t0 (float): time window lower bound
            tf (float): time window upper bound
            r0 (tuple): position vector at t0
            rf (tuple): position vector at tf
            v0 (tuple): velocity vector at t0
            vf (tuple): velocity vector at tf
        """
        self.t0 = t0
        self.tf = tf
        self.r0 = r0
        self.rf = rf
        self.v0 = v0
        self.vf = vf
        return

    def solve(self):
        """Minimize two-impulse rendezvous deltaV."""
        # Define Pygmo problem
        params = (self.t0, self.tf, self.r0, self.rf, self.v0, self.vf)
        prob = pg.problem(udp=deltav_udp(params))
        print(prob)

        # Define Pygmo algorithm
        
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





    

    