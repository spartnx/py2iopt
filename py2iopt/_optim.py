"""
Functions to solve the optimization.
"""

import pykep as pk
import numpy as np
from numba import njit

from ._state_transition import psi, psi_inv

class deltav_udp:
    """User-Defined Problem formulating the deltaV minimization problem.
    
    Args:
        params (tuple): t0, tf, r0, rf, v0, vf, mu
    """
    def __init__(self, params):
        self.params = params
        return

    def fitness(self, x):
        # Add constraint x[0] <= x[1]
        return [obj_fcn(x, *self.params)]

    def get_bounds(self):
        t0 = self.params[0]
        tf = self.params[1]
        lbs = np.full((2,), t0)
        ubs = np.full((2,), tf)
        return (lbs, ubs)

    def gradient(self, x):
        return grad_fcn(x, *self.params)


@njit
def obj_fcn(x, t0, tf, r0, rf, v0, vf, mu):
    """Compute deltaV of two-impulse rendezvous with initial and final coasting.
    
    Args:
        x (2-tuple): times of the first and second impulses
        t0 (float): time window lower bound
        tf (float): time window upper bound
        r0 (tuple): position vector at t0
        rf (tuple): position vector at tf
        v0 (tuple): velocity vector at t0
        vf (tuple): velocity vector at tf
        mu (float): central body's gravitational parameter

    Returns:
        (float): magnitude of the maneuver's deltaV
    """
    # Propagate orbit forward from t0 to x[0]
    r1, v1m = pk.propagate_lagrangian(r0=r0, v0=v0, tof=x[0]-t0, mu=mu)
    # Propagate orbit backward from tf to x[1]
    r2, v2p = pk.propagate_lagrangian(r0=rf, v0=vf, tof=x[1]-tf, mu=mu)
    # Solve Lambert problem between (x[0], r1) and (x[1], r2)
    model = pk.lambert_problem(r1=r1, r2=r2, tof=x[1]-x[0], mu=mu, max_revs=0, cw=False)
    # Retrieve initial and final velocity vectors along the transfer trajectory (m/s)
    v1p = np.array(model.get_v1()[0])
    v2m = np.array(model.get_v2()[0])
    # Compute the impulses' deltaVs
    v1m = np.array(v1m)
    v2p = np.array(v2p)
    dv1_mag = np.linalg.norm(v1p - v1m)
    dv2_mag = np.linalg.norm(v2p - v2m)
    return dv1_mag + dv2_mag


@njit
def grad_fcn(x, t0, tf, r0, rf, v0, vf, mu):
    """Compute the gradient of the deltaV function as per Primer Vector Theory.
    
    Args:
        x (2-tuple): times of the first and second impulses
        t0 (float): time window lower bound
        tf (float): time window upper bound
        r0 (tuple): position vector at t0
        rf (tuple): position vector at tf
        v0 (tuple): velocity vector at t0
        vf (tuple): velocity vector at tf
        mu (float): central body's gravitational parameter

    Returns:
        (tuple): gradient of the maneuver's deltaV function w.r.t. x
    """
    # Steps to compute the deltaV vectors
    # Propagate orbit forward from t0 to x[0]
    r1, v1m = pk.propagate_lagrangian(r0=r0, v0=v0, tof=x[0]-t0, mu=mu)
    # Propagate orbit backward from tf to x[1]
    r2, v2p = pk.propagate_lagrangian(r0=rf, v0=vf, tof=x[1]-tf, mu=mu)
    # Solve Lambert problem between (x[0], r1) and (x[1], r2)
    model = pk.lambert_problem(r1=r1, r2=r2, tof=x[1]-x[0], mu=mu, max_revs=0, cw=False)
    # Retrieve initial and final velocity vectors along the transfer trajectory (m/s)
    v1p = np.array(model.get_v1()[0])
    v2m = np.array(model.get_v2()[0])
    # Compute the impulses' deltaVs
    v1m = np.array(v1m)
    v2p = np.array(v2p)
    dv1 = v1p - v1m
    dv2 = v2p - v2m
    dv1_mag = np.linalg.norm(dv1)
    dv2_mag = np.linalg.norm(dv2)

    # Steps to compute the gradient of the deltaV function as per Primer Vector Theory
    # Orbital elements
    sma,ecc,_,_,_,ea0 = pk.ic2par(r1, v1p, mu=mu)
    _,_,_,_,_,eaf = pk.ic2par(r2, v2m, mu=mu)
    slr = abs(sma*(1-ecc**2))
    if ecc < 1: # ea = eccentric anomaly
        ta0 = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(ea0/2))
        taf = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(eaf/2))
    else: # ea = Guadermannian
        ta0 = 2*np.arctan(((1+ecc)/(ecc-1))**.5 * np.tan(ea0/2))
        taf = 2*np.arctan(((1+ecc)/(ecc-1))**.5 * np.tan(eaf/2))
    # Angular momentum vector
    h = np.cross(np.array(r1), np.array(v1p))
    # Compute state-transition matrix between t_start and t_end, i.e., stm(t_start, t_end)
    psi_inv_0 = psi_inv(x[0], r1, v1p, h, ecc, slr, ta0, mu)
    psi_f = psi(x[1], r2, v2m, h, ecc, slr, taf, mu)
    stm = psi_f @ psi_inv_0
    # Compute initial time derivative of the primer vector
    p0 = (dv1 / dv1_mag).reshape((3,1))
    pf = (dv2 / dv2_mag).reshape((3,1))
    stm_11 = stm[:3,:3]
    stm_12 = stm[:3,3:6]
    pdot0 = np.linalg.inv(stm_12) @ (pf - stm_11 @ p0)
    pdot0_scalar = (pdot0.T @ p0).item(0)
    # Compute final time derivative of the primer vector 
    stm_21 = stm[3:6,:3]
    stm_22 = stm[3:6,3:6]
    pdotf = stm_21 @ p0 + stm_22 @ pdot0
    pdotf_scalar = (pdotf.T @ pf).item(0)
    return (-pdot0_scalar * dv1_mag, -pdotf_scalar * dv2_mag)

def time_constraint():
    return