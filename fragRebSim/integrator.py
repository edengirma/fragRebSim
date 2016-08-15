# Standard python modules
import random as rnd
from math import cos, sin, sqrt

# These modules need to be pip installed.
import numpy as np
import rebound
from astropy import units as u
# For Jupyter/IPython notebook
from tqdm import tnrange

# From fragRebSim
from .dMdEdist import dMdEdist
from .forces import sim_constants as sc
from .forces import bulge_force, cluster_force, disk_force, halo_force
from .funcs import beta_dist, mstar_dist, r_tidal, rstar_func


class RebSimIntegrator:

    def __init__(self, Nstars, Nfrag):
        self.Nstars = Nstars
        self.Nfrag = Nfrag
        self.Nout = 10000
        self.max_time = 1.0e7
        self.posx = [[[] for y in range(Nfrag)] for x in range(Nstars)]
        self.posy = [[[] for y in range(Nfrag)] for x in range(Nstars)]
        self.posz = [[[] for y in range(Nfrag)] for x in range(Nstars)]
        self.dmde = dMdEdist()
        self.forces = []

    def set_Nstars(self, new_Nstars):
        self.Nstars = new_Nstars

    def set_Nfrag(self, new_Nfrag):
        self.Nfrag = new_Nfrag

    def set_Nout(self, new_Nout):
        self.Nout = new_Nout

    # Galaxy potential,
    # from http://adsabs.harvard.edu/abs/2014ApJ...793..122K
    # Note: There is a typo in that paper where "a_d" is said to be
    # 2750 kpc, it should be 2.75 kpc.
    def migrationAccel(self, reb_sim):
        ps = reb_sim.contents.particles
        x2 = ps[1].x**2
        y2 = ps[1].y**2
        z2 = ps[1].z**2
        r = sqrt(x2 + y2 + z2)
        rho2 = x2 + y2
        zbd = sqrt(z2 + sc.bd**2)

        ps[1].ax += cluster_force(r, ps[1].x) + bulge_force(r, ps[1].x) +\
            disk_force(r, ps[1].x, rho2, zbd) + halo_force(r, ps[1].x)
        ps[1].ay += cluster_force(r, ps[1].y) + bulge_force(r, ps[1].y) +\
            disk_force(r, ps[1].y, rho2, zbd) + halo_force(r, ps[1].y)
        ps[1].az += cluster_force(r, ps[1].z) + bulge_force(r, ps[1].z) +\
            disk_force(r, ps[1].z, rho2, zbd) + halo_force(r, ps[1].z)

    def sim_integrate(self):
        m_hole = sc.m_hole
        star_masses = []
        star_radii = []
        tidal_radii = []

        for star in tnrange(self.Nstars, desc='Star', leave=False):
            # Randomly drawn mass of star
            xstar = rnd.random()
            m_star = mstar_dist(xstar)
            star_masses.append(m_star)

            # Determined radius of star
            r_star = rstar_func(m_star) * sc.RsuntoAU
            star_radii.append(r_star)
            # Distance spread for fragments
            rads = [r_star * float(f)/float(self.Nfrag+1)
                    for f in range(self.Nfrag+1)]
            rads.pop(0)

            # Determined tidal radius of star
            r_t = r_tidal(m_star, r_star)
            tidal_radii.append(r_t)

            # Set position of star; random sphere point picking
            u1 = rnd.uniform(-1.0, 1.0)
            th1 = rnd.uniform(0., 2. * np.pi)
            star_direc = np.array([sqrt(1.0 - (u1)**2) * cos(th1),
                                   sqrt(1.0 - (u1)**2) * sin(th1),
                                   u1])
            star_vec = [r_t * d for d in star_direc]

            # Binding energy spread, with beta value randomly drawn from
            # beta distribution
            xbeta = rnd.random()
            beta = beta_dist(xbeta)
            NRGs = self.dmde.energy_spread(beta, self.Nfrag)

            # Converted NRGs list from cgs to proper units
            pi = np.pi
            natural_u = (u.AU / (u.yr / (2.0 * pi)))**2
            nrg_scale = ((r_star * sc.AUtoRsun)**(-1.0) * (m_star)**(2.0 / 3.0)
                         * (m_hole / 1.0e6)**(1.0 / 3.0))
            energies = [(nrg_scale * nrg *
                         (u.cm / u.second)**2).to(natural_u).value
                        for nrg in NRGs]

            # Calculating velocities
            vels = [sqrt((2.0 * g) + (2 * m_hole / r_t)) for g in energies]

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

            for fi, frag in enumerate(tnrange(self.Nfrag,
                                              desc='Fragment', leave=False)):

                # Velocity vector of fragment
                vel = vels[frag]
                frag_velvec = [vel * v / n for v in velocity_vec]
                # Position vector of Fragment
                rad = rads[frag]
                frag_posvec = [(r_t + rad) * p for p in star_direc]

                # Set up rebound simulation
                reb_sim = rebound.Simulation()
                reb_sim.integrator = "ias15"
                reb_sim.add(m=m_hole)
                reb_sim.dt = 1.0e-15

                # Add particle to rebound simulation
                reb_sim.add(m=0.0, x=frag_posvec[0], y=frag_posvec[1],
                            z=frag_posvec[2], vx=frag_velvec[0],
                            vy=frag_velvec[1], vz=frag_velvec[2])
                reb_sim.N_active = 1
                reb_sim.additional_forces = self.migrationAccel
                reb_sim.force_is_velocity_dependent = 1
                reb_sim.exit_max_distance = 15.0 * sc.scale  # 15 pc in AU
                ps = reb_sim.particles

                stop = np.log10(self.max_time)
                times = np.logspace(-17.0, stop, self.Nout)
                times = np.insert(times, 0.0, 0)
                for ti, time in enumerate(times):
                    try:
                        reb_sim.integrate(time, exact_finish_time=1)
                        self.posx[star][frag].append(ps[1].x / sc.scale)
                        self.posy[star][frag].append(ps[1].y / sc.scale)
                        self.posz[star][frag].append(ps[1].z / sc.scale)
                    except rebound.Escape as error:
                        print(error)
                        break

                    # Semi-major axis criterion:
                    # Cuts particles closely bound to black hole
                    if (2.0 * ps[1].a / sc.scale > 0.0 and
                            2.0 * ps[1].a / sc.scale < 1.0):
                        break
