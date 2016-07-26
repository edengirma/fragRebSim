# Standard python modules
import random as rnd
from math import cos, sin, sqrt

# These modules need to be pip installed.
import numpy as np
import rebound
from astropy import constants as const

# For Jupyter/IPython notebook
from tqdm import tnrange

# From frag-rebsim-pkg
from .dMdEdist import beta_mass_interp, energy_spread
from .forces import sim_constants as sc
from .funcs import migrationAccel, mstar_dist, r_tidal, rstar_func, beta_dist

if __name__ == '__main__':
    Nstars = int(raw_input('Number of stars: '))
    nfrag = int(raw_input('Number of fragments per star: '))
    positions = sim_integrate(Nstars, nfrag)


# ---------------------------------- INTEGRATING --------------------------
def sim_integrate(Nstars, nfrag):
    Nout = 100

    posx = [[[] for y in range(nfrag)] for x in range(Nstars)]
    posy = [[[] for y in range(nfrag)] for x in range(Nstars)]
    posz = [[[] for y in range(nfrag)] for x in range(Nstars)]

    star_masses = []
    star_radii = []
    tidal_radii = []
    sphere_points = []

    m_hole = sc.m_hole

    for star in tnrange(Nstars, desc='Star', leave=False):
        # Randomly drawn mass of star
        xstar = rnd.random()
        m_star = mstar_dist(xstar)
        star_masses.append(m_star)

        # Determined radius of star
        r_star = const.R_sun.to('AU').value * rstar_func(m_star)
        star_radii.append(r_star)

        # Determined tidal radius of star
        r_t = r_tidal(m_star, r_star)
        tidal_radii.append(r_t)

        # Set position of star; random sphere point picking
        u1 = rnd.uniform(-1.0, 1.0)
        th1 = rnd.uniform(0., 2. * np.pi)
        star_vec = np.array([r_t * sqrt(1.0 - (u1)**2) * cos(th1),
                             r_t * sqrt(1.0 - (u1)**2) * sin(th1),
                             r_t * u1])  # star is a distance r_t from hole
        sphere_points.append(star_vec)

        # Distances of fragments from tidal radius
        rads = [r_star * float(f) / float(nfrag + 1) for f in range(nfrag + 1)]
        rads.pop(0)

        # Binding energy spread, from beta value randomly draw from
        # beta distribution
        xbeta = rnd.random()
        beta = beta_dist(xbeta)
        energies = energy_spread(beta, nfrag)

        # Randomly draw velocity vector direction
        phi2 = rnd.uniform(0., 2. * np.pi)

        x = star_vec[0]
        y = star_vec[1]
        z = star_vec[2]
        r = np.linalg.norm(star_vec)

        randomvelvec = [
            (x * (r - z + z * cos(phi2)) - r * y * sin(phi2)) /
            (r**2 * sqrt(2.0 - 2.0 * z / r)),
            (y * (r - z + z * cos(phi2)) + r * x * sin(phi2)) /
            (r**2 * sqrt(2.0 - 2.0 * z / r)),
            ((r - z) * z - (x**2 + y**2) * cos(phi2)) /
            (r**2 * sqrt(2.0 - 2.0 * z / r))
        ]

        velocity_vec = np.cross(star_vec, randomvelvec)
        n = np.linalg.norm(velocity_vec)

        for frag in tnrange(nfrag, desc='Fragment', leave=False):

            # Finalize velocity vector of fragment
            vel = vels[frag]
            frag_velvec = [vel * v / n for v in velocity_vec]

            # Set up simulation
            sim = rebound.Simulation()
            sim.integrator = "ias15"
            sim.add(m=m_hole)

            # Add particle
            sim.add(m=0.0, x=star_vec[0], y=star_vec[1], z=star_vec[2],
                    vx=frag_velvec[0], vy=frag_velvec[1], vz=frag_velvec[2])
            sim.N_active = 1
            sim.additional_forces = migrationAccel
            sim.force_is_velocity_dependent = 1
            ps = sim.particles

            times = np.linspace(0.0, 1.0e9 * 2.0 * np.pi, Nout)
            for ti, time in enumerate(times):
                sim.integrate(time, exact_finish_time=1)
                posx[star][frag].append(ps[1].x / sc.scale)
                posy[star][frag].append(ps[1].y / sc.scale)
                posz[star][frag].append(ps[1].z / sc.scale)

                if np.linalg.norm([ps[1].x, ps[1].y, ps[1].z]) / sc.scale > 20:
                    break

                if 0 < 2 * ps[1].a / sc.scale and 2 * ps[1].a / sc.scale < 5.0:
                    break

    return [posx, posy, posz]

print('integrator imported')
