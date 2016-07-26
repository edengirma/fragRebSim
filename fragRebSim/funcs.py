from math import sqrt

from .forces import (bulge_force, cluster_force, disk_force, halo_force,
                     sim_constants)


# Galaxy potential, from http://adsabs.harvard.edu/abs/2014ApJ...793..122K
# Note: There is a typo in that paper where "a_d" is said to be 2750 kpc,
# it should be 2.75 kpc.
def migrationAccel(reb_sim):
    ps = reb_sim.particles

    x2 = ps[1].x**2
    y2 = ps[1].y**2
    z2 = ps[1].z**2

    r = sqrt(x2 + y2 + z2)
    rho2 = x2 + y2
    zbd = sqrt(z2 + sim_constants.bd**2)

    ps[1].ax += cluster_force(r, ps[1].x) + bulge_force(r, ps[1].x) +\
        disk_force(r, ps[1].x, rho2, zbd) + halo_force(r, ps[1].x)
    ps[1].ay += cluster_force(r, ps[1].x) + bulge_force(r, ps[1].y) +\
        disk_force(r, ps[1].y, rho2, zbd) + halo_force(r, ps[1].y)
    ps[1].az += cluster_force(r, ps[1].x) + bulge_force(r, ps[1].z) +\
        disk_force(r, ps[1].z, rho2, zbd) + halo_force(r, ps[1].z)


# Randomly draw stellar mass, derived from Salpeter's mass function:
# Xi(m) =  m**(-2.35)
# http://mathworld.wolfram.com/RandomNumber.html provides an expression for
# the proper variate.
# Returns mass between the bounds [x0,x1] in solar masses when inputting a
# probability 0 <= x <= 1.
def mstar_dist(y):
    n = -2.35
    x0 = 100.0
    x1 = 0.1
    return ((x1**(n + 1.) - x0**(n + 1.)) * y + x0**(n + 1.))**(1. / (n + 1.))


# Star radius determined via mass-radius power law, from
# http://faculty.buffalostate.edu/sabatojs/courses/GES639/S10/
# reading/mass_luminosity.pdf:
def rstar_func(m_star):
    return m_star**0.8


# Beta distribution function
def beta_dist(y):
    n = -2
    x0 = 2.5
    x1 = 0.5
    return ((x1**(n + 1.) - x0**(n + 1.)) * y + x0**(n + 1.))**(1. / (n + 1.))


# Tidal radius for a star
def r_tidal(m_star, r_star):
    return r_star * (sim_constants.m_hole / m_star)**(1.0 / 3.0)
