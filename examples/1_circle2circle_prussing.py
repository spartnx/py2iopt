"""
An example drawn from Chapter 2 written by J. E. Prussing in 
"Spacecraft Trajectory Optimization" edited by B. A. Conway.

The goal of this script is to validate pyi2opt by reproducing
fig. 2.5 and fig. 2.6 using the Ipopt algorithm.
"""

import numpy as np
import pykep as pk
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from py2iopt import TwoImpulseRDV

if __name__ == "__main__":
    # Inputs (nondimensional)
    mu = 1
    t0 = 0*(2*np.pi)
    tf = 0.9*(2*np.pi)
    rc0 = (1, 0, 0) # chaser's initial position
    rt0 = (0, 1.6, 0) # target's initial position
    vc0 = (0, (mu/1)**.5, 0) # chaser's initial velocity
    vt0 = (-(mu/1.6)**.5, 0, 0) # target's initial velocity
    rcf, vcf = pk.propagate_lagrangian(r0=rt0, v0=vt0, tof=tf-t0, mu=mu)

    # Initialize TwoImpulseRDV object
    tirdv = TwoImpulseRDV(mu=mu, verbosity=1)
    tirdv.set_problem(t0, tf, rc0, rcf, vc0, vcf)
    tirdv.pretty_settings()

    # Solve problem
    tirdv.solve()

    # Display results
    tirdv.pretty_results(time_scale=2*np.pi)

    # Plot trajectory and primer vector magnitude history
    tirdv.plot(plot_optimal=False, time_scale=2*np.pi) # plot Lambert maneuver
    tirdv.plot(plot_optimal=True, time_scale=2*np.pi) # plot solution with initial and/or final coasting
    plt.show()


