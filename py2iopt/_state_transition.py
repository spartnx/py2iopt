"""
Functions to compute the time-evolution of the Primer Vector.

These functions compute the 6-by-6 matrices psi(t) and psi_inv(t) 
used to compute the state transition matrix stm(t,t0).
The matrix stm(t,t0) is needed to compute the primer vector p(t) and its time 
derivative p_dot(t) over time between the initial and final times of the maneuver.
     _          _                 _           _
    |    p(t)    |  =  stm(t,t0) |    p(t0)    |
    |_ p_dot(t) _|               |_ p_dot(t0) _|

    stm(t,t0) = psi(t) * psi_inv(t0)


For more details on how to compute stm(t,t0) from psi(t) and psi(t0), refer to:

    Glandorf, D. R., "Lagrange Multipliers and the State Transition Matrix for Coasting Arc,"
    Journal of Optimization Theory and Application, Vol. 15, No. 5, 1975
"""

from numba import njit
import numpy as np

@njit
def psi(t, r, v, h, ecc, slr, ta, mu):
    """Compute the matrix psi.

    Args:
        t (float): time
        r (3-tuple): position vector
        r (3-tuple): velocity vector
        h (3-tuple): angular momentum 
        ecc (float): eccentricity
        slr (float): semi-latus rectum
        ta (float): true anomaly
        mu (float): central body's gravitational parameter

    Returns:
        (array): matrix psi
    """
    # Matrix parameters
    r = np.array(r).reshape((3,1))
    v = np.array(v).reshape((3,1))
    h = np.array(h).reshape((3,1))
    r_mag = np.linalg.norm(r)
    h_mag = np.linalg.norm(h)
    q = (-slr/h_mag)*(slr + r_mag)*r_mag*np.cos(ta)
    s = ((slr/h_mag)*(slr+r_mag)*r_mag*np.sin(ta) - 3*ecc*slr*t) / (1-ecc**2)
    f1 = r_mag*np.cos(ta)
    f2 = r_mag*np.sin(ta)
    f3 = -(h_mag/slr)*np.sin(ta)
    f4 = (h_mag/slr)*(ecc+np.cos(ta))
    f5 = mu / r_mag**3
    f6 = 3*t
    f7 = 3*mu*t/r_mag**3
    f8 = f3 + s*f5
    f9 = f4 + q*f5

    # Matrix
    mat = np.concatenate((
                    np.concatenate((f1*r-s*v, f8*r-f1*v), axis=0),
                    np.concatenate((f2*r-q*v, f9*r-f2*v), axis=0),
                    np.concatenate((2*r-f6*v, f7*r-v   ), axis=0),
                    np.concatenate((v       , -f5*r    ), axis=0),
                    np.concatenate((f1*h    , f3*h     ), axis=0),
                    np.concatenate((f2*h    , f4*h     ), axis=0)
                ), axis=1)
    return mat

@njit
def psi_inv(t, r, v, h, ecc, slr, ta, mu):
    """Compute the inverse of the matrix psi.

    Args:
        t (float): time
        r (3-tuple): position vector
        r (3-tuple): velocity vector
        h (3-tuple): angular momentum 
        ecc (float): eccentricity
        slr (float): semi-latus rectum
        ta (float): true anomaly
        mu (float): central body's gravitational parameter

    Returns:
        (array): inverse of the matrix psi
    """
    # Matrix parameters
    r = np.array(r)
    v = np.array(v)
    h = np.array(h)
    sigma = np.cross(r, h).reshape((1,3))
    w = np.cross(v, h).reshape((1,3))
    r_mag = np.linalg.norm(r)
    h_mag = np.linalg.norm(h)
    q = (-slr/h_mag)*(slr + r_mag)*r_mag*np.cos(ta)
    s = ((slr/h_mag)*(slr+r_mag)*r_mag*np.sin(ta) - 3*ecc*slr*t) / (1-ecc**2)
    f1 = r_mag*np.cos(ta)
    f2 = r_mag*np.sin(ta)
    f3 = -(h_mag/slr)*np.sin(ta)
    f4 = (h_mag/slr)*(ecc+np.cos(ta))
    f5 = mu / r_mag**3
    f6 = 3*t
    b1 = h_mag + f5*(f1*q - f2*s)
    b2 = f6*h_mag + f3*q - f4*s
    b3 = f6*h_mag + 2*(f3*q - f4*s)
    b4 = f1*q - f2*s

    # Matrix
    h = h.reshape((1,3))
    mat_inv = (1/h_mag**3) * np.concatenate((
                                    np.concatenate((f2*f5*sigma - f4*w, 2*f4*sigma - f2*w), axis=1),
                                    np.concatenate((-f1*f5*sigma+f3*w , -2*f3*sigma+f1*w ), axis=1),
                                    np.concatenate((h_mag*w           , -h_mag*sigma     ), axis=1),
                                    np.concatenate((-b1*sigma + b2*w  , -b3*sigma + b4*w ), axis=1),
                                    np.concatenate((f4*h              , -f2*h            ), axis=1),
                                    np.concatenate((-f3*h             , f1*h             ), axis=1)
                                ), axis=0)
    return mat_inv
