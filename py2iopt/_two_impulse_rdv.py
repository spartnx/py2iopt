"""
Functions made available to user to optimize two-impulse rendezvous trajectories.
"""

import pygmo as pg

from ._optim import deltav_udp

class TwoImpulseRDV:
    def __init__(self, mu=1, algo="ipopt", log_freq=10):
        """Define dynamics environment and algorithm.
        
        Args:
            mu (float): central body's gravitational parameter
            algo (str): optimization algorithm
            log_freq (int): frequency at which to display algorithm iterations 
        """
        self.mu = mu
        self.algo = algo
        self.log_freq = log_freq
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
        prob = pg.problem(udp=deltav_udp(params, self.algo))
        print(prob)

        # Define Pygmo algorithm
        if self.algo == "ipopt":
            algo = pg.algorithm(pg.ipopt())
        elif self.algo == "l-bfgs-b":
            algo = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))
        algo.set_verbosity(self.log_freq)

        # Define Pygmo population and initialize it at [self.t0, self.tf]
        pop = pg.population(prob)
        pop.push_back(x = [self.t0, self.tf])

        # Minimize deltaV
        pop = algo.evolve(pop)
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





    

    