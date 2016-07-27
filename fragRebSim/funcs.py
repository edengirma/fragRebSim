from .forces import sim_constants


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
