"""
This script tests the good behavior of TwoImpulseRDV class on ellipse-to-inclined-ellipse rendezvous maneuvers.
"""

import numpy as np
import pykep as pk
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from py2iopt import TwoImpulseRDV, obj_grad, traj_and_pvec_data

if __name__ == "__main__":
    run_loop = 1
    mu = pk.MU_EARTH # m^3/s^2
    t0 = 0 # s
    DU = 42164*1000 # canonical distance unit, m

   # Orbital elements of chaser's orbit
    r1 = DU # semi-major axis, m
    e1 = 0.1 # eccentricity
    inc1 = np.radians(5) # inclination, rad
    raan1 = np.radians(234) # Right Ascension of the Ascending Node, rad
    w1 = np.radians(78) # argument of periapse, rad
    E1 = 0 # initial eccentric anomaly (same as true anomaly for a circular orbit), rad

    # Orbital elements of target's orbit
    r2 = 0.7*DU # semi-major axis, m
    e2 = 0.3 # eccentricity
    inc2 = np.radians(45) # inclination, rad
    raan2 = np.radians(60) # Right Ascension of the Ascending Node, rad
    w2 = np.radians(145) # argument of periapse, rad
    E2 = 0 # initial eccentric anomaly, rad

    flight_times = np.array([1, 2, 5, 10, 15]) # days
    sep_angle = np.linspace(10, 360, endpoint=False, num=35) # chaser-to-target separation angle, degrees
    R0, V0 = pk.par2ic([r1, e1, inc1, raan1, w1, E1], mu) # chaser's orbital state
    
    if run_loop == 1:
        tirdv_ipopt = TwoImpulseRDV(mu=mu, verbosity=0)

        flight_times = np.array([1, 2, 5, 10, 15]) # days
        sep_angle = np.linspace(10, 360, endpoint=False, num=35) # chaser-to-target separation angle, degrees
        deltaVs_lambert = {tf: [] for tf in flight_times}
        deltaVs_primer_ipopt = {tf: [] for tf in flight_times}
        primer_grad_t1 = {tf: [] for tf in flight_times}
        primer_grad_t2 = {tf: [] for tf in flight_times}
        max_primer = {tf: [] for tf in flight_times}
        for tof in flight_times:
            tf = float(tof*24*3600) # seconds
            for theta in sep_angle:
                print(f"TOF = {tof}, theta = {theta}")
                E2 = float(np.radians(theta)) # radians
                Rt0, Vt0 = pk.par2ic([r2, e2, inc2, raan2, w2, E2], mu)
                Rf, Vf = pk.propagate_lagrangian(r0=Rt0, v0=Vt0, tof=tf-t0, mu=mu)

                model = pk.lambert_problem(r1=R0, r2=Rf, tof=tf-t0, mu=mu, max_revs=0, cw=False)
                V_p = list(model.get_v1())[0]
                V_m = list(model.get_v2())[0]
                dV_lambert0 = np.array(V_p) - np.array(V0)
                dV_lambertf = np.array(Vf) - np.array(V_m)
                dV_lambert = np.linalg.norm(dV_lambert0) + np.linalg.norm(dV_lambertf)

                tirdv_ipopt.set_problem(t0, tf, R0, Rf, V0, Vf)
                tirdv_ipopt.solve()
                dV_ipopt = tirdv_ipopt.deltav

                deltaVs_lambert[tof].append(dV_lambert)
                deltaVs_primer_ipopt[tof].append(dV_ipopt)

                primer_grad = obj_grad((t0,tf), t0, tf, R0, Rf, V0, Vf, mu)
                primer_grad_t1[tof].append(primer_grad[0])
                primer_grad_t2[tof].append(primer_grad[1])
                arcs = []
                nm1th_arc = (t0, tf)
                data = traj_and_pvec_data(t0, tf, R0, Rf, V0, Vf, arcs, nm1th_arc, mu, N_pts=1e3)
                _, _, _, _, _, _, _, _, primer = data
                max_primer[tof].append(max(primer))


        deltaVs_all = {"Lambert": pd.DataFrame(data=deltaVs_lambert, index=sep_angle),
                       "Primer Ipopt": pd.DataFrame(deltaVs_primer_ipopt, index=sep_angle)}

        primer = {"Gradient wrt t1": pd.DataFrame(data=primer_grad_t1, index=sep_angle),
                  "Gradient wrt t2": pd.DataFrame(data=primer_grad_t2, index=sep_angle),
                  "Max primer": pd.DataFrame(data=max_primer, index=sep_angle)}

        with pd.ExcelWriter('5_ellipse2inclined_analysis_files\ellipse-to-inclined-ellipse.xlsx') as writer:  
            deltaVs_all["Lambert"].to_excel(writer, sheet_name='Lambert')
            deltaVs_all["Primer Ipopt"].to_excel(writer, sheet_name='Ipopt')

        with pd.ExcelWriter('5_ellipse2inclined_analysis_files\ellipse-to-inclined-ellipse_gradient_max-primer.xlsx') as writer:  
            primer["Gradient wrt t1"].to_excel(writer, sheet_name="Gradient wrt t1")
            primer["Gradient wrt t2"].to_excel(writer, sheet_name="Gradient wrt t2")
            primer["Max primer"].to_excel(writer, sheet_name="Max primer")

    else:
        # Plot figures
        tf = 15 *(24*3600)
        E2 = np.radians(160) # radians
        Rt0, Vt0 = pk.par2ic([r2, e2, inc2, raan2, w2, E2], mu)
        Rf, Vf = pk.propagate_lagrangian(r0=Rt0, v0=Vt0, tof=tf-t0, mu=mu)

        tirdv_ipopt = TwoImpulseRDV(mu=mu, verbosity=0)
        tirdv_ipopt.set_problem(t0, tf, R0, Rf, V0, Vf)
        tirdv_ipopt.solve()
        tirdv_ipopt.plot(plot_optimal=True)
        tirdv_ipopt.plot(plot_optimal=False)
        plt.show()
