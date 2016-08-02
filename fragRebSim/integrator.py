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
        self.Nout = 1000
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
        # t = reb_sim.contents.t
        x2 = ps[1].x**2
        y2 = ps[1].y**2
        z2 = ps[1].z**2
        # bh_force_array = [ps[1].ax, ps[1].ay, ps[1].az]
        r = sqrt(x2 + y2 + z2)
        rho2 = x2 + y2
        zbd = sqrt(z2 + sc.bd**2)

        ps[1].ax += cluster_force(r, ps[1].x) + bulge_force(r, ps[1].x) +\
            disk_force(r, ps[1].x, rho2, zbd) + halo_force(r, ps[1].x)
        ps[1].ay += cluster_force(r, ps[1].y) + bulge_force(r, ps[1].y) +\
            disk_force(r, ps[1].y, rho2, zbd) + halo_force(r, ps[1].y)
        ps[1].az += cluster_force(r, ps[1].z) + bulge_force(r, ps[1].z) +\
            disk_force(r, ps[1].z, rho2, zbd) + halo_force(r, ps[1].z)

        # cluster_force_array = [cluster_force(r, ps[1].x),
        #                        cluster_force(r, ps[1].y),
        #                        cluster_force(r, ps[1].z)]
        # bulge_force_array = [bulge_force(r, ps[1].x),
        #                      bulge_force(r, ps[1].y),
        #                      bulge_force(r, ps[1].z)]
        # disk_force_array = [disk_force(r, ps[1].x, rho2, zbd),
        #                     disk_force(r, ps[1].y, rho2, zbd),
        #                     disk_force(r, ps[1].z, rho2, zbd)]
        # halo_force_array = [halo_force(r, ps[1].x),
        #                     halo_force(r, ps[1].y),
        #                     halo_force(r, ps[1].z)]
        # force_vector = [cluster_force_array, bulge_force_array,
        #                 disk_force_array, halo_force_array, bh_force_array]
        # positions = [ps[1].x, ps[1].y, ps[1].z]
        # velocities = [ps[1].vx, ps[1].vy, ps[1].vz]
        # accelerations = [ps[1].ax, ps[1].ay, ps[1].az]
        # point = [t, force_vector, positions, velocities, accelerations]
        # self.forces.append(point)

    def sim_integrate(self):
        m_hole = sc.m_hole
        star_masses = []
        star_radii = []
        tidal_radii = []
        sphere_points = []

        for star in tnrange(self.Nstars, desc='Star', leave=False):
            # Randomly drawn mass of star
            xstar = rnd.random()
            m_star = mstar_dist(xstar)
            star_masses.append(m_star)

            # Determined radius of star
            r_star = rstar_func(m_star)*sc.RsuntoAU
            star_radii.append(r_star)

            # Determined tidal radius of star
            r_t = r_tidal(m_star, r_star)
            tidal_radii.append(r_t)

            # Set position of star; random sphere point picking
            u1 = rnd.uniform(-1.0, 1.0)
            th1 = rnd.uniform(0., 2. * np.pi)
            star_vec = np.array([r_t * sqrt(1.0 - (u1)**2) * cos(th1),
                                 r_t * sqrt(1.0 - (u1)**2) * sin(th1),
                                 r_t * u1])  # star is r_t from hole
            sphere_points.append(star_vec)

            # Binding energy spread, from beta value randomly draw from
            # beta distribution
            xbeta = rnd.random()
            beta = beta_dist(xbeta)
            NRGs = self.dmde.energy_spread(beta, self.Nfrag)

            # Converted NRGs list from cgs to proper units
            pi = np.pi
            natural_u = (u.AU/(u.yr/(2.0 * pi)))**2
            nrg_scale = ((r_star * sc.AUtoRsun)**(-1.0) * (m_star)**(2.0/3.0) *
                         (m_hole/1.0e6)**(1.0/3.0))
            print(nrg_scale)
            energies = [(nrg_scale * nrg *
                        (u.cm/u.second)**2).to(natural_u).value
                        for nrg in NRGs]

            # Calculating excess velocities
            velinfs = [sqrt(2.0 * x) for x in energies]
            vels = [sqrt((2.0 * nrg)+(2 * m_hole / r_t)) for nrg in energies]
            # vels = [sqrt((2.0 * m_hole / r_t)-(2.0 * m_hole/(sc.rc/2.0))
            # + (2.0 * nrg)) for nrg in energies]

            # Debugging
            # print('Mass of star(solMass): ', m_star, '\n')
            # print('Radius of star (solRad): ', r_star*sc.Rsun_AU, '\n')
            # print('Tidal radius of star: ', r_t, '\n')
            # print('Excess velocities: ', velinfs, '\n')
            # print('Vels due to orbit: ', sqrt((2.0 * m_hole / r_t)), '\n')
            print('Total velocities: ', vels)
            # sys.exit()

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

                # Total velocity magnitude of fragment
                vel = vels[frag]
                vel_units = u.AU/(u.yr/(2.0 * pi))
                # Excess velocity of fragment
                frag_velinf = velinfs[fi]
                # Velocity criterion for integrated particles
                # if (frag_velinf > ((2500. * u.km / u.second)
                #                    .to(vel_units).value)):
                #     continue

                frag_velvec = [vel * v / n for v in velocity_vec]

                # Set up rebound simulation
                reb_sim = rebound.Simulation()
                reb_sim.integrator = "ias15"
                reb_sim.add(m=m_hole)
                reb_sim.dt = 1.0e-15

                # Add particle to rebound simulation
                reb_sim.add(m=0.0, x=star_vec[0], y=star_vec[1], z=star_vec[2],
                            vx=frag_velvec[0], vy=frag_velvec[1],
                            vz=frag_velvec[2])
                reb_sim.N_active = 1
                reb_sim.additional_forces = self.migrationAccel
                reb_sim.force_is_velocity_dependent = 1
                ps = reb_sim.particles

                times = np.linspace(0.0, self.max_time, self.Nout)
                for ti, time in enumerate(times):
                    reb_sim.integrate(time, exact_finish_time=1)
                    self.posx[star][frag].append(ps[1].x / sc.scale)
                    self.posy[star][frag].append(ps[1].y / sc.scale)
                    self.posz[star][frag].append(ps[1].z / sc.scale)
                    # Add additional distance criterion
                    # if (np.linalg.norm([
                    #         self.ps[1].x, self.ps[1].y,
                    #         self.ps[1].z])/sc.scale > 20):
                    #     break

                    if (2.0*ps[1].a/sc.scale > 0.0 and
                            2.0*ps[1].a/sc.scale < 5.0):
                        break

        print('Masses:', star_masses)
