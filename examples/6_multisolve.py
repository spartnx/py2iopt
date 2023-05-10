"""
This script tests the multi_solve() method which allows users to solve 
in parallel several two-impulse rendezvous maneuvers with different 
initial guesses.

This is an attempt to "global-ify" the optimization procedure 
implemented in the TwoImpulseRDV class.

Comaprison with fig. 5b from following reference:

    Luo, Y-Z., Zhang, J., Li, H-Y., and Tang, G-J., "Interactive Optimization
    Approach for Optimal Impulsive Rendezvous Using Primer Vector and Evolutionary
    Algorithms," Acta Astronautica 67 (2010) 396-405.

The multi_solve() method finds the same solution as Fig. 5b in Luo et al.
"""

import numpy as np
import pykep as pk
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from py2iopt import TwoImpulseRDV


if __name__ == "__main__":
    # Inputs (nondimensional)
    mu = pk.MU_EARTH # m^3/s^2
    r = (6378 + 400)*1000 # circular orbit radius, m
    period = 2*np.pi*np.sqrt(r**3/mu) # circular orbit period, s
    t0 = 0 # s
    tf = 2.3*period # s
    rc0 = (r, 0, 0) # chaser's initial position, m
    rt0 = (-r, 0, 0) # target's initial position, m
    vc0 = (0, (mu/r)**.5, 0) # chaser's initial velocity, m/s
    vt0 = (0, -(mu/r)**.5, 0) # target's initial velocity. m/s
    rcf, vcf = pk.propagate_lagrangian(r0=rt0, v0=vt0, tof=tf-t0, mu=mu)

    # Initialize TwoImpulseRDV object
    tirdv = TwoImpulseRDV(mu=mu, verbosity=0)
    tirdv.set_problem(t0, tf, rc0, rcf, vc0, vcf)
    tirdv.pretty_settings()

    # Initial guesses
    x0 = [[t0, tf],
          [1763, 10070]] # < from Luo et al. (Table 2)

    # Solve problem
    tirdv.multi_solve(x0=x0)

    # Display results
    tirdv.pretty_results()

    # Plot trajectory and primer vector magnitude history
    tirdv.plot(plot_optimal=False) # plot Lambert maneuver
    tirdv.plot(plot_optimal=True) # plot solution with initial and/or final coasting
    plt.show()


