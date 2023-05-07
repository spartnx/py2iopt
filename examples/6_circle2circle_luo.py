"""
An example drawn from:

    Luo, Y-Z., Zhang, J., Li, H-Y., and Tang, G-J., "Interactive Optimization Approach for Impulse rendezvous
    Using Primer Vector and Evolutionary Algorithms," Acta Astronautice, 67, pp. 396-405, 2010

The goal of this script is to validate pyi2opt by reproducing
fig. fig. 5a and fig. 5b using the Ipopt algorithm.

Fig 5a replicated but not fig. 5b. Ipopt cannot reduce the deltaV.
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
    tf = 2.3*(2*np.pi)
    rc0 = (1, 0, 0) # chaser's initial position
    rt0 = (-1, 0, 0) # target's initial position
    vc0 = (0, (mu/1)**.5, 0) # chaser's initial velocity
    vt0 = (0, -(mu/1)**.5, 0) # target's initial velocity
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
    tirdv.plot(plot_optimal=True, time_scale=2*np.pi) # plot solution with initial and/or finalcoasting
    plt.show()


