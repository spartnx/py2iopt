"""
Functions made available to user to optimize two-impulse rendezvous trajectories.
"""

import pygmo as pg
import numpy as np
import pykep as pk
import matplotlib.pyplot as plt

from ._optim import deltav_udp
from ._plot_helpers import traj_and_pvec_data

class TwoImpulseRDV:
    def __init__(self, mu=1, algo="ipopt", verbosity=10):
        """Define dynamics environment and algorithm.
        
        Args:
            mu (float): central body's gravitational parameter
            algo (str): optimization algorithm
            verbosity (int): algorithm verbosity level

        exitcode:
            0 : problem initialized but not yet attempted
            1 : problem successfully solved
            -1 : problem failed to be solved
        """
        self.mu = mu
        self.algo = algo
        self.verbosity = verbosity
        self.exitcode = 0
        self.ready = False
        return


    def set_problem(self, t0, tf, r0, rf, v0, vf):
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
        self.ready = True
        return


    def solve(self):
        """Minimize two-impulse rendezvous deltaV."""
        assert self.ready == True, "Please first call `set_problem()`"
        # Outputs' placeholers
        self.t1 = None # first impulse time
        self.t2 = None # second impulse time
        self.deltav = None # rendezvous deltaV

        # Define Pygmo problem
        params = (self.t0, self.tf, self.r0, self.rf, self.v0, self.vf, self.mu)
        prob = pg.problem(udp=deltav_udp(params, self.algo))
        print(prob)

        # Define Pygmo algorithm
        if self.algo == "ipopt":
            uda = pg.ipopt()
        elif self.algo == "l-bfgs-b":
            uda = pg.scipy_optimize(method="L-BFGS-B")
        algo = pg.algorithm(uda)
        algo.set_verbosity(self.verbosity)

        # Define Pygmo population and initialize it at [self.t0, self.tf]
        pop = pg.population(prob)
        pop.push_back(x = np.array([self.t0, self.tf]))

        # Minimize deltaV
        pop = algo.evolve(pop)

        # Check successful optimization termination
        if self.algo in ["ipopt"]:
            if uda.get_last_opt_result == 0:
                print("Solver did not converge (ipopt)")
                self.exitcode = -1
            else:
                print("Problem successfully solved (ipopt)")
                self.exitcode = 1
        elif self.algo in ["l-bfgs-b"]:
            self.exitcode = 1
            # if uda.success == True:
            #     print("Solver did not converge (l-bfgs-b)")
            #     self.exitcode = -1
            # else:
            #     print("Problem successfully solved (l-bfgs-b)")
            #     self.exitcode = 1

        # Record optimal solution
        if self.exitcode == 1:
            self.t1 = pop.champion_x[0]
            self.t2 = pop.champion_x[1] 
            self.deltav = pop.champion_f[0] 
        return


    def plot(self, plot_optimal=True, time_scale=1, units=1, N_pts=1e3):
        """Plot 3D rendezvous trajectory and history of primer vector magnitude."""
        if plot_optimal==True and self.exitcode==1:
            arcs = []
            nm1th_arc = (self.t1, self.t2, 0, 0)
            fig_title = "Optimal maneuver (with coasting)"
        elif plot_optimal==False:
            arcs = []
            nm1th_arc = (self.t0, self.tf, 0, 0)
            fig_title = "Lambert maneuver (no coasting)"

        # compute data to plot
        data = traj_and_pvec_data(self.t0, 
                                  self.tf, 
                                  self.r0, 
                                  self.rf, 
                                  self.v0, 
                                  self.vf, 
                                  arcs, 
                                  nm1th_arc, 
                                  self.mu, 
                                  N_pts=N_pts)

        arcs_data, pvec_mag, impulse_time, primer, position, time, radius_vector, dV_tot = data

        # Plot the history of the magnitude of the primer vector over time
        fig = plt.figure()
        fig.suptitle(fig_title)
        ax1 = fig.add_subplot(121)
        for i in range(len(arcs_data)):
            t = np.array(time[i]) / time_scale
            p = pvec_mag[i]
            ax1.plot(t, p, zorder=1)
        for i in range(len(arcs)+2):
            t = impulse_time[i]/time_scale
            p = np.linalg.norm(primer[i])
            ax1.scatter(t, p, color='k', marker='x', s=20, zorder=2)
        ax1.grid(True)
        ax1.set(xlabel="Time", ylabel="Primer magnitude") 
        ax1.set_title("Magnitude of the primer vector w.r.t. time\n(dV = " + str(round(dV_tot,4)) + ")")

        # Plot the trajectory
        sma0,_,_,_,_,_ = pk.ic2par(self.r0, self.v0, mu=self.mu)
        smaf,_,_,_,_,_ = pk.ic2par(self.rf, self.vf, mu=self.mu)
        T0 = 2*np.pi*(sma0**3/self.mu)**.5
        Tf = 2*np.pi*(smaf**3/self.mu)**.5
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.scatter([0], [0], [0], s=10, color=['g'], label="Central body")
        ax2.scatter([self.r0[0]], [self.r0[1]], [self.r0[2]], s=40, color=['r'], label="Initial position")
        ax2.scatter([self.rf[0]], [self.rf[1]], [self.rf[2]], s=40, color=['b'], label="Final position")
        for pos in position:
            ax2.scatter([pos[0]], [pos[1]], [pos[2]], color=['k'], marker='x')
        pk.orbit_plots.plot_kepler(self.r0, self.v0, tof=T0, mu=self.mu, N=2000, units=units, axes=ax2, color="0.7", label="Initial orbit")
        pk.orbit_plots.plot_kepler(self.rf, self.vf, tof=Tf, mu=self.mu, N=2000, units=units, axes=ax2, color="0.5", label="Final orbit")

        for i in range(len(arcs_data)):
            X = [r[0] for r in radius_vector[i]]
            Y = [r[1] for r in radius_vector[i]]
            Z = [r[2] for r in radius_vector[i]]
            ax2.plot(X, Y, Z, label='Arc ' + str(i))
        ax2.legend()
        ax2.set_title("Rendezvous trajectory")
        ax2.axis("equal")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z") 
        return fig


    def pretty_settings(self):
        """Display of the settings"""
        print(f"\nGravitational parameter : {self.mu}")
        print(f"Algorithm               : {self.algo}")
        print(f"Verbosity               : {self.verbosity}")
        return


    def pretty_results(self, time_scale=1):
        """Display of the outputs"""
        print(f"\nExit code : {self.exitcode}")
        print(f"t1        : {round(self.t1/time_scale,4)}")
        print(f"t2        : {round(self.t2/time_scale,4)}")
        print(f"deltaV    : {round(self.deltav,4)}")
        return 


