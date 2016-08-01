import rebound
import numpy as np
def migrationAccel(reb_sim):
    ps = reb_sim.contents.particles
    t = reb_sim.contents.t
    print(t, ps[1].x)

reb_sim = rebound.Simulation()
reb_sim.integrator = "ias15"
reb_sim.add(m=1.)

# Add particle to rebound simulation
reb_sim.add(m=0.0, a=1.);
reb_sim.N_active = 1
reb_sim.additional_forces = migrationAccel
s = reb_sim.particles

times = np.linspace(0.0, 1., 10)
for ti, time in enumerate(times):
    reb_sim.integrate(time, exact_finish_time=1)
