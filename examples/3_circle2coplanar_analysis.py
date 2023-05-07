"""
This script tests the good behavior of TwoImpulseRDV class on circle-to-coplanar-circle rendezvous maneuvers.
In particular, the 'ipopt' and 'l-bfgs-b' algorithms are compared.
"""

import numpy as np
import pykep as pk
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from py2iopt import TwoImpulseRDV, lambert_solver, obj_grad, traj_and_pvec_data

if __name__ == "__main__":
    run_loop = 0
    mu = pk.MU_EARTH # m^3/s^2
    t0 = 0 # s
    r0 = 42164*1000 # orbital radius, m
    r1 = 36000*1000 # final orbital radius, m
    v0 = (mu/r0)**.5 # orbital speed, m/s
    v1 = (mu/r1)**.5 # orbital speed, m/s   
    R0 = (r0, 0, 0) # initial position vector of the chaser, m 
    V0 = (0, v0, 0) # initial velocity vector of the chaser, m/s
    DU = r0 # canonical distance unit, m

    if run_loop == 1:
        tirdv_ipopt = TwoImpulseRDV(mu=mu, algo="ipopt", verbosity=0)
        tirdv_lbfgsb = TwoImpulseRDV(mu=mu, algo="l-bfgs-b", verbosity=0)

        flight_times = np.array([1, 2, 5, 10, 15]) # days
        sep_angle = np.linspace(10, 360, endpoint=False, num=35) # chaser-to-target separation angle, degrees
        deltaVs_lambert = {tf: [] for tf in flight_times}
        deltaVs_primer_lbfgs = {tf: [] for tf in flight_times}
        deltaVs_primer_ipopt = {tf: [] for tf in flight_times}
        primer_grad_t1 = {tf: [] for tf in flight_times}
        primer_grad_t2 = {tf: [] for tf in flight_times}
        max_primer = {tf: [] for tf in flight_times}
        for tof in flight_times:
            tf = float(tof*24*3600) # seconds
            for theta in sep_angle:
                print(f"TOF = {tof}, theta = {theta}")
                theta_rad = float(np.radians(theta)) # radians
                Rt0 = (r1*np.cos(theta_rad), r1*np.sin(theta_rad), 0) # initial position vector of the target, m
                Vt0 = (-v1*np.sin(theta_rad), v1*np.cos(theta_rad), 0) # initial velocity vector of the target, m/s
                Rf, Vf = pk.propagate_lagrangian(r0=Rt0, v0=Vt0, tof=tf-t0, mu=mu)

                _,_,_, dV0_list, dVf_list,_,_,_,_,_,_ = lambert_solver(0, R0, Rf, V0, Vf, t0, tf, mu, plot=False, units=1000, tof=100*3600)
                tirdv_ipopt.set_problem(t0, tf, R0, Rf, V0, Vf)
                tirdv_lbfgsb.set_problem(t0, tf, R0, Rf, V0, Vf)
                tirdv_ipopt.solve()
                tirdv_lbfgsb.solve()
                dV_ipopt = tirdv_ipopt.deltav
                dV_lbfgs = tirdv_lbfgsb.deltav

                deltaVs_lambert[tof].append(np.linalg.norm(np.array(dVf_list[0])) + np.linalg.norm(np.array(dV0_list[0])))
                deltaVs_primer_ipopt[tof].append(dV_ipopt)
                deltaVs_primer_lbfgs[tof].append(dV_lbfgs)

                primer_grad = obj_grad((t0,tf), t0, tf, R0, Rf, V0, Vf, mu)
                primer_grad_t1[tof].append(primer_grad[0])
                primer_grad_t2[tof].append(primer_grad[1])
                arcs = []
                nm1th_arc = (t0, tf, 0, 0)
                data = traj_and_pvec_data(t0, tf, R0, Rf, V0, Vf, arcs, nm1th_arc, mu, N_pts=1e3)
                _, _, _, _, _, _, _, _, primer = data
                max_primer[tof].append(max(primer))


        deltaVs_all = {"Lambert": pd.DataFrame(data=deltaVs_lambert, index=sep_angle),
                       "Primer L-BFGS-B": pd.DataFrame(deltaVs_primer_lbfgs, index=sep_angle),
                       "Primer Ipopt": pd.DataFrame(deltaVs_primer_ipopt, index=sep_angle)}

        primer = {"Gradient wrt t1": pd.DataFrame(data=primer_grad_t1, index=sep_angle),
                  "Gradient wrt t2": pd.DataFrame(data=primer_grad_t2, index=sep_angle),
                  "Max primer": pd.DataFrame(data=max_primer, index=sep_angle)}

        with pd.ExcelWriter('3_circle2coplanar_analysis_files\circle-to-coplanar-circle_deltaV.xlsx') as writer:  
            deltaVs_all["Lambert"].to_excel(writer, sheet_name='Lambert')
            deltaVs_all["Primer L-BFGS-B"].to_excel(writer, sheet_name='L-BFGS-B')
            deltaVs_all["Primer Ipopt"].to_excel(writer, sheet_name='Ipopt')

        with pd.ExcelWriter('3_circle2coplanar_analysis_files\circle-to-coplanar-circle_gradient_max-primer.xlsx') as writer:  
            primer["Gradient wrt t1"].to_excel(writer, sheet_name="Gradient wrt t1")
            primer["Gradient wrt t2"].to_excel(writer, sheet_name="Gradient wrt t2")
            primer["Max primer"].to_excel(writer, sheet_name="Max primer")

    # Plot figures
    theta_rad = np.radians(260)
    tf = 5 *(24*3600)
    Rt0 = (r1*np.cos(theta_rad), r1*np.sin(theta_rad), 0) # initial position vector of the target, m
    Vt0 = (-v1*np.sin(theta_rad), v1*np.cos(theta_rad), 0) # initial velocity vector of the target, m/s
    Rf, Vf = pk.propagate_lagrangian(r0=Rt0, v0=Vt0, tof=tf-t0, mu=mu)

    tirdv_ipopt = TwoImpulseRDV(mu=mu, algo="ipopt", verbosity=0)
    tirdv_ipopt.set_problem(t0, tf, R0, Rf, V0, Vf)
    tirdv_ipopt.solve()
    tirdv_ipopt.plot(plot_optimal=True)
    tirdv_ipopt.plot(plot_optimal=False)

    tirdv_lbfgs = TwoImpulseRDV(mu=mu, algo="l-bfgs-b", verbosity=0)
    tirdv_lbfgs.set_problem(t0, tf, R0, Rf, V0, Vf)
    tirdv_lbfgs.solve()
    tirdv_lbfgs.plot(plot_optimal=True)
    tirdv_lbfgs.plot(plot_optimal=False)
    plt.show()
