import pykep as pk
import numpy as np
import math

from ._state_transition import psi, psi_inv


def traj_and_pvec_data(t0, tf, R0, Rf, V0, Vf, arcs, nm1th_arc, mu, N_pts=1e3):
    """Compute the data needed to plot the trajectory and the primer vector magnitude history.

    Args:
        t0 (float): time window lower bound
        tf (float): time window upper bound
        R0 (tuple): position vector at t0
        Rf (tuple): position vector at tf
        V0 (tuple): velocity vector at t0
        Vf (tuple): velocity vector at tf 
        arcs (list): tuples in the format (ti, dVi) = (ith impulse time, ith deltaV vector) 
        nm1th_arc (tuple): times of the (n-1)th and nth impulses (n = number of impulses)
        mu: central body's gravitational parameter
        n_pts: number of points to compute trajectory and magnitude of primer vector over time

    Returns:
        arcs_data
        primer_vector_magnitude_2
        impulse_time
        primer
        position
        time_2
        radius_vector_2
        dV_tot
        primer_vector_magnitude
    """
    impulse_time = []
    position = []
    velocity_p = [] # V1_p, V2_p, ..., V(n-1)_p, V(n)_p
    velocity_m = [] # V1_m, V2_m, ..., V(n-1)_m, V(n)_m
    arcs_data = [] 
    primer = []
    primer_dot_0 = []
    primer_dot_f = []
    dV_tot = 0

    # Propagate initial orbit between t0 and t1 (if initial coasting)
    # Collect R1
    # Collect V1_m
    # Arc 0: (t0, R0, V0, t1, R1, V1_m)
    if len(arcs) > 0:
        t1 = arcs[0][0]
    else:
        t1 = nm1th_arc[0]

    if t1 == t0:
        initial_coasting = False
        R1 = R0
        V1_m = V0
    else:
        initial_coasting = True
        # Propagate orbit forward from t0 to t1
        # Compute position and velocity at the end with the primer vector directly
        R1, V1_m = pk.propagate_lagrangian(r0=R0, v0=V0, tof=t1-t0, mu=mu)

    position.append(R1)
    velocity_m.append(V1_m)
    if t1 > t0:
        arcs_data.append((t0, R0, V0, t1, R1, V1_m))


    # Propagate final orbit backward from tf to tn (if final coasting)
    # Collect Rn
    # Collect Vn_p
    # Arc n: (tn, Rn, Vn_p, tf, Rf, Vf)
    tn = nm1th_arc[1]
    if tn == tf: 
        final_coasting = False
        Rn = Rf
        Vn_p = Vf
    else:
        final_coasting = True
        # Propagate orbit backward from tf to tn
        # Compute position and velocity at the end with the primer vector directly
        Rn, Vn_p = pk.propagate_lagrangian(r0=Rf, v0=Vf, tof=tn-tf, mu=mu)


    # Compute the initial velocity using the given deltaV and propagate arcs 1 to n-2 (if arcs list is non-empty; if empty, skip this part and the (n-1)th arc becomes arc 1)
    # Compute the initial value of the primer vector over each arc from the given deltaV
    # Collect R2, R3, ..., R(n-1)
    # Collect V1_p, V2_p, ..., V(n-2)_p
    # Collect V2_m, V3_m, ..., V(n-2)_m, V(n-1)_m
    # Collect p1, p2, ..., p(n-2)
    # Arc 1: (t1, R1, V1_p, t2, R2, V2_m)
    # Arc j: (tj, Rj, Vj_p, t(j+1), R(j+1), V(j+1)_m)
    for i in range(len(arcs)):
        t_start = arcs[i][0]
        if i == len(arcs) - 1:
            t_end = nm1th_arc[0]
        else:
            t_end = arcs[i+1][0]
        V_preimpulse = velocity_m[-1]
        dV = arcs[i][1]
        V_start = tuple(V_preimpulse[i] + dV[i] for i in range(3))
        R_start = position[-1]
        p_start = np.array(dV).reshape((3,1)) / np.linalg.norm(np.array(dV))
        R_end, V_end = pk.propagate_lagrangian(r0=R_start, v0=V_start, tof=t_end-t_start, mu=mu)

        impulse_time.append(t_start)
        position.append(R_end)
        velocity_m.append(V_end)
        velocity_p.append(V_start)
        arcs_data.append((t_start, R_start, V_start, t_end, R_end, V_end))
        primer.append(p_start) 
        dV_tot += np.linalg.norm(dV)


    # Solve Lambert's problem for the (n-1)th arc given R(n-1) and Rn
    # Collect V(n-1)_p, Vn_m
    # Collect dV(n-1) = V(n-1)_p - V(n-1)_m
    # Collect dVn = Vn_p - Vn_m
    # Collect p(n-1), pn
    # Arc n-1: (t(n-1), R(n-1), V(n-1)_p, tn, Rn, Vn_m)
    t_start = nm1th_arc[0]
    t_end = nm1th_arc[1]
    R_start = position[-1]
    R_end = Rn
    V_start = velocity_m[-1]
    V_end = Vn_p
    model = pk.lambert_problem(r1=R_start, r2=R_end, tof=t_end-t_start, mu=mu, max_revs=0, cw=False)
    V_p = list(model.get_v1())[0]
    V_m = list(model.get_v2())[0]
    dV_start = (np.array(V_p) - np.array(V_start)).reshape((3,1))
    dV_end = (np.array(V_end) - np.array(V_m)).reshape((3,1))
    p_start = dV_start / np.linalg.norm(dV_start)
    p_end = dV_end / np.linalg.norm(dV_end)

    impulse_time += [t_start, t_end]
    position.append(Rn)
    velocity_m.append(V_m)
    velocity_p += [V_p, Vn_p]
    if t_end < tf:
        arcs_data += [(t_start, position[-2], velocity_p[-2], t_end, position[-1], velocity_m[-1]), (t_end, position[-1], velocity_p[-1], tf, Rf, Vf)]
    else:
        arcs_data += [(t_start, position[-2], velocity_p[-2], t_end, position[-1], velocity_m[-1])]
    primer += [p_start, p_end]
    dV_tot += np.linalg.norm(dV_start) + np.linalg.norm(dV_end)


    # Compute position, velocity, and primer vector magnitude for arcs 1 to n-1 by using the initial and final boundary values of the primer vector to solve the differential equation
    if initial_coasting == True and final_coasting == True:
        interior_arcs = arcs_data[1:-1]
    elif initial_coasting == True and final_coasting == False:
        interior_arcs = arcs_data[1:]
    elif initial_coasting == False and final_coasting == True:
        interior_arcs = arcs_data[0:-1]
    else:
        interior_arcs = arcs_data
    primer_vector_magnitude = []
    time = []
    radius_vector = []
    velocity_vector = []
    primer_vector_magnitude_2 = []
    time_2 = []
    radius_vector_2 = []
    velocity_vector_2 = []
    for i in range(len(interior_arcs)):
        # Initial and final conditions of the arc
        t_start = interior_arcs[i][0]
        R_start = interior_arcs[i][1]
        V_start = interior_arcs[i][2]
        t_end = interior_arcs[i][3]
        R_end = interior_arcs[i][4]
        V_end = interior_arcs[i][5]
        # Orbital elements
        sma,ecc,_,_,_,E0 = pk.ic2par(R_start, V_start, mu=mu)
        _,_,_,_,_,Ef = pk.ic2par(R_end, V_end, mu=mu)
        slr = sma*(1-ecc**2)
        f_start = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(E0/2))
        f_end = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(Ef/2))
        # Angular momentum vector
        H = np.cross(np.array(R_start), np.array(V_start))
        # Compute state-transition matrix between t_start and t_end, i.e., PHI(t_start, t_end)
        LAMBDA_INV_0 = psi_inv(t_start, 
                                np.array(R_start, dtype=np.float64), 
                                np.array(V_start, dtype=np.float64), 
                                np.array(H, dtype=np.float64), 
                                np.array(H, dtype=np.float64).reshape((1,3)), 
                                ecc, slr, f_start, mu
                            )
        LAMBDA_f = psi(t_end, 
                        np.array(R_end, dtype=np.float64).reshape((3,1)), 
                        np.array(V_end, dtype=np.float64).reshape((3,1)), 
                        np.array(H, dtype=np.float64).reshape((3,1)), 
                        ecc, slr, f_end, mu
                    )
        PHI = LAMBDA_f @ LAMBDA_INV_0
        # Compute initial time derivative of the primer vector
        p_end = primer[i+1]
        p_start = primer[i]
        PHI_11 = PHI[:3,:3]
        PHI_12 = PHI[:3,3:6]
        pdot_start = np.linalg.inv(PHI_12) @ (p_end - PHI_11 @ p_start)
        primer_dot_0.append(pdot_start)
        # Compute final time derivative of the primer vector 
        PHI_21 = PHI[3:6,:3]
        PHI_22 = PHI[3:6,3:6]
        pdot_end = PHI_21 @ p_start + PHI_22 @ pdot_start
        primer_dot_f.append(pdot_end)
        # Compute the history of the magnitude of the primer vector between t_start and t_end
        primer_state0 = np.concatenate((p_start, pdot_start), axis=0)
        step = (t_end - t_start) / N_pts
        time_arc = [t_start + step*j for j in range(int(N_pts))]
        primer_vector_magnitude_arc = []
        radius_vector_arc = []
        velocity_vector_arc = []
        for t in time_arc:
            R, V = pk.propagate_lagrangian(r0=R_start, v0=V_start, tof=t-t_start, mu=mu)
            radius_vector_arc.append(R)
            velocity_vector_arc.append(V)
            sma,ecc,_,_,_,E = pk.ic2par(R, V, mu=mu)
            slr = sma*(1-ecc**2)
            f = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(E/2))
            LAMBDA = psi(t, 
                            np.array(R, dtype=np.float64).reshape((3,1)), 
                            np.array(V, dtype=np.float64).reshape((3,1)), 
                            np.array(H, dtype=np.float64).reshape((3,1)), 
                            ecc, slr, f, mu
                        )
            PHI = LAMBDA @ LAMBDA_INV_0
            primer_vector = PHI[:3,:6] @ primer_state0
            primer_vector_magnitude_arc.append(np.linalg.norm(primer_vector))
        time += time_arc
        primer_vector_magnitude += primer_vector_magnitude_arc
        radius_vector += radius_vector_arc
        velocity_vector += velocity_vector_arc
        time_2.append(time_arc)
        primer_vector_magnitude_2.append(primer_vector_magnitude_arc)
        radius_vector_2.append(radius_vector_arc)
        velocity_vector_2.append(velocity_vector_arc)


    # Compute position, velocity, and the primer vector magnitude history over arc 0 by enforcing p1, p1_dot
    if initial_coasting:
        t_start = arcs_data[0][0]
        R_start = arcs_data[0][1]
        V_start = arcs_data[0][2]
        t_end = arcs_data[0][3]
        R_end = arcs_data[0][4]
        V_end = arcs_data[0][5]
        # Orbital elements
        sma,ecc,_,_,_,Ef = pk.ic2par(R_end, V_end, mu=mu)
        slr = sma*(1-ecc**2)
        if ecc < 1e-3 or math.isnan(Ef):
            Ef = 0
        f_end = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(Ef/2))
        # Angular momentum vector 
        H = np.cross(np.array(R_end), np.array(V_end))
        # Compute the constant part of the state-transition matrix at t_end, i.e., P_INV(t_end)
        LAMBDA_INV_f = psi_inv(t_end, 
                                np.array(R_end, dtype=np.float64), 
                                np.array(V_end, dtype=np.float64), 
                                np.array(H, dtype=np.float64), 
                                np.array(H, dtype=np.float64).reshape((1,3)), 
                                ecc, slr, f_end, mu
                            )
        # Compute time series
        p_end = primer[0]
        pdot_end = primer_dot_0[0]  
        primer_statef = np.concatenate((p_end, pdot_end), axis=0)
        step = (t_end - t_start) / N_pts
        time_arc = [t_start + step*j for j in range(int(N_pts))]
        primer_vector_magnitude_arc = []
        radius_vector_arc = []
        velocity_vector_arc = []
        for t in time_arc:
            R, V = pk.propagate_lagrangian(r0=R_end, v0=V_end, tof=t-t_end, mu=mu)
            radius_vector_arc.append(R)
            velocity_vector_arc.append(V)
            sma,ecc,_,_,_,E = pk.ic2par(R, V, mu=mu)
            if ecc <= 1e-3 or math.isnan(E):
                mean_motion = (mu / sma**3)**.5
                E = Ef + (t - t_end) * mean_motion 
                E %= 2*np.pi
                if E >= np.pi and E <= 2*np.pi:
                    E = E - 2*np.pi
            slr = sma*(1-ecc**2)
            f = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(E/2))
            LAMBDA = psi(t, 
                            np.array(R, dtype=np.float64).reshape((3,1)), 
                            np.array(V, dtype=np.float64).reshape((3,1)), 
                            np.array(H, dtype=np.float64).reshape((3,1)), 
                            ecc, slr, f, mu
                        )
            PHI = LAMBDA @ LAMBDA_INV_f
            primer_vector = PHI[:3,:6] @ primer_statef
            primer_vector_magnitude_arc.append(np.linalg.norm(primer_vector))
        time = time_arc + time
        primer_vector_magnitude = primer_vector_magnitude_arc + primer_vector_magnitude
        radius_vector = radius_vector_arc + radius_vector
        velocity_vector = velocity_vector_arc + velocity_vector
        time_2.insert(0, time_arc)
        primer_vector_magnitude_2.insert(0, primer_vector_magnitude_arc)
        radius_vector_2.insert(0, radius_vector_arc)
        velocity_vector_2.insert(0, velocity_vector_arc)


    # Compute position, velocity, and the primer vector magnitude history over arc n by enforcing pn, pn_dot
    if final_coasting:
        t_start = arcs_data[-1][0]
        R_start = arcs_data[-1][1]
        V_start = arcs_data[-1][2]
        t_end = arcs_data[-1][3]
        R_end = arcs_data[-1][4]
        V_end = arcs_data[-1][5]
        # Orbital elements
        sma,ecc,_,_,_,E0 = pk.ic2par(R_start, V_start, mu=mu)
        if ecc < 1e-3 or math.isnan(E0):
            E0 = 0
        slr = sma*(1-ecc**2)
        f_start = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(E0/2))
        # Angular momentum vector 
        H = np.cross(np.array(R_start), np.array(V_start))
        # Compute the constant part of the state-transition matrix at t_start, i.e., P_INV(t_start)
        LAMBDA_INV_0 = psi_inv(t_start, 
                                np.array(R_start, dtype=np.float64), 
                                np.array(V_start, dtype=np.float64), 
                                np.array(H, dtype=np.float64), 
                                np.array(H, dtype=np.float64).reshape((1,3)), 
                                ecc, slr, f_start, mu
                            )
        # Compute time series
        p_start = primer[-1]
        pdot_start = primer_dot_f[-1]  
        primer_state0 = np.concatenate((p_start, pdot_start), axis=0)
        step = (t_end - t_start) / N_pts
        time_arc = [t_start + step*j for j in range(int(N_pts))]
        primer_vector_magnitude_arc = []
        radius_vector_arc = []
        velocity_vector_arc = []
        for t in time_arc:
            R, V = pk.propagate_lagrangian(r0=R_start, v0=V_start, tof=t-t_start, mu=mu)
            radius_vector_arc.append(R)
            velocity_vector_arc.append(V)
            sma,ecc,_,_,_,E = pk.ic2par(R, V, mu=mu)
            if ecc <= 1e-3 or math.isnan(E):
                mean_motion = (mu / sma**3)**.5
                E = E0 + (t - t_start) * mean_motion 
                E %= 2*np.pi
                if E >= np.pi and E <= 2*np.pi:
                    E = E - 2*np.pi
            slr = sma*(1-ecc**2)
            f = 2*np.arctan(((1+ecc)/(1-ecc))**.5 * np.tan(E/2))
            LAMBDA = psi(t, 
                            np.array(R, dtype=np.float64).reshape((3,1)), 
                            np.array(V, dtype=np.float64).reshape((3,1)), 
                            np.array(H, dtype=np.float64).reshape((3,1)), 
                            ecc, slr, f, mu
                        )
            PHI = LAMBDA @ LAMBDA_INV_0
            primer_vector = PHI[:3,:6] @ primer_state0
            primer_vector_magnitude_arc.append(np.linalg.norm(primer_vector))
        time += time_arc
        primer_vector_magnitude += primer_vector_magnitude_arc
        radius_vector += radius_vector_arc
        velocity_vector += velocity_vector_arc
        time_2.append(time_arc)
        primer_vector_magnitude_2.append(primer_vector_magnitude_arc)
        radius_vector_2.append(radius_vector_arc)
        velocity_vector_2.append(velocity_vector_arc)

    return arcs_data, primer_vector_magnitude_2, impulse_time, primer, position, time_2, radius_vector_2, dV_tot, primer_vector_magnitude



