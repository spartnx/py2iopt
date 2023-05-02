"""
Functions to solve the optimization.
"""

import pykep as pk
import pygmo as pg

from ._state_transition import psi, psi_inv

class deltav_udp:
    """UDP defining the deltaV as fitness"""
    def __init__(self):
        return

def obj_fcn():
    """Where deltaV is computed"""
    return

def grad_fcn():
    """Where the gradient of the objective function is the defined,
    in accordance with Primer Vector Theory"""
    return