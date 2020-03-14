from sympy.stats.crv import SingleContinuousPSpace
from sympy.stats.drv import SingleDiscretePSpace
from sympy.stats.rv import random_symbols
from sympy import sqrt

def create_compound(sym, dist):
    comp_dist = compound_rv_map[dist.compound_distribution.__class__.__name__](sym, dist)
    return comp_dist if comp_dist is not None else dist

compound_rv_map = {
    'NormalDistribution': lambda sym, dist: compute_normal(sym, dist),
    'PoissonDistribution': lambda sym, dist: compute_poisson(sym, dist),
    'ExponentialDistribution': lambda sym, dist: compute_exponential(sym, dist),
    'GammaDistribution': lambda sym, dist: compute_gamma(sym, dist)
}

def compute_normal(sym ,dist):
    mean, std = dist.compound_distribution.args
    if random_symbols(mean):
        from sympy.stats.crv_types import NormalDistribution
        if isinstance(mean.pspace.distribution, NormalDistribution):
            mu, sigma = mean.pspace.distribution.args
            return SingleContinuousPSpace(sym, NormalDistribution(mu, sqrt(sigma ** 2 + std ** 2)))
    else:
        from sympy.stats.crv_types import RayleighDistribution, LaplaceDistribution
        if isinstance(std.pspace.distribution, RayleighDistribution):
            sigma = std.pspace.distribution.args[0]
            return SingleContinuousPSpace(sym, LaplaceDistribution(mean, sigma))

def compute_poisson(sym, dist):
    from sympy.stats.drv_types import NegativeBinomialDistribution
    from sympy.stats.crv_types import GammaDistribution
    rate = dist.compound_distribution.args[0]
    if isinstance(rate.pspace.distribution, GammaDistribution):
        k, theta = rate.pspace.distribution.args
        return SingleDiscretePSpace(sym, NegativeBinomialDistribution(k, theta/(1+theta)))
