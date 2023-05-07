# py2iopt
A Python implementation of optimal two-impulse rendezvous maneuvers in accordance with Primer Vector Theory. A Lambert rendezvous maneuver is improved by the addition of an initial and/or final coasting phase. The trajectories are strictly keplerian (no perturbation) and the Lambert solver does not allow for multi-revolution transfers. The optimization algorithm used in `py2iopt` is `Ipopt`.

### Dependencies

- `pykep`, `numpy`, `matplotilb`, `numba`, `pygmo`

### Basic usage
Examples on how to use `py2iopt` are included in `./examples/`. Example `1_` is a replication of Fig. 2.5 and Fig. 2.6 from Chapter 2 of "Spacecraft Trajectory Optimization" edited by B. A. Conway. Examples `2_` to `5_` compute families of two-impulse rendezvous maneuvers for a variety of relative positions between the chaser and the target.

First start by importing the `py2iopt` and `pykep`.

```python
#import sys
#sys.path.append("../")  # make sure the pyrqlaw folder is exposed
import py2iopt
import pykep as pk
```

Define the central body's gravitational parameter, and lower and upper bounds of the time window:

```python
mu = 1
t0 = 0*(2*np.pi)
tf = 0.9*(2*np.pi)
```

Define the chaser's initial position and velocity vectors.

```python
rc0 = (1, 0, 0) # chaser's initial position
vc0 = (0, (mu/1)**.5, 0) # chaser's initial velocity
```

Define the target's initial position and velocity vectors, and propagate them from `t0` to `tf`.

```python
rt0 = (0, 1.6, 0) # target's initial position
vt0 = (-(mu/1.6)**.5, 0, 0) # target's initial velocity
rcf, vcf = pk.propagate_lagrangian(r0=rt0, v0=vt0, tof=tf-t0, mu=mu)
```

Initialize the `TwoImpulseRDV` class instance with the previously-defined inputs.

```python
tirdv = TwoImpulseRDV(mu=mu, verbosity=1)
tirdv.set_problem(t0, tf, rc0, rcf, vc0, vcf)
```

Solve the problem.

```python
tirdv.solve()
```

Finally, display the results and plot the rendezvous maneuvers with and without coasting phases.

```python
Exit code : 1
Coasting  : True
t1        : 0.2207
t2        : 0.9
deltaV    : 0.2146
```

<p align="center">
  <img src="./plots//circle-to-coplanar-circle-rendezvous-without-coasting.PNG" width="400" title="RDV without coasting">
</p>
<p align="center">
<em>Example of a rendezvous maneuver without coasting phase (deltaV = 0.3747).</em>
</p>

<p align="center">
  <img src="./plots//circle-to-coplanar-circle-rendezvous-with-coasting.PNG" width="400" title="RDV with coasting">
</p>
<p align="center">
<em>Example of a rendezvous maneuver with an initial coasting phase (deltaV = 0.2146).</em>
</p>

### Important notes
- loclally optimal solutions
