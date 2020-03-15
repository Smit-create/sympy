from sympy import Symbol, Basic, Lambda, Symbol, S, sympify
from sympy.stats.crv import ContinuousDistributionHandmade, SingleContinuousPSpace, ContinuousDistribution
from sympy.stats.drv import DiscreteDistributionHandmade, SingleDiscretePSpace, DiscreteDistribution
from sympy.stats.joint_rv import JointDistributionHandmade, JointPSpace, JointDistribution
from sympy.stats.frv import SingleFinitePSpace, SingleFiniteDistribution
from sympy.stats.frv_types import FiniteDistributionHandmade
from sympy.stats.rv import PSpace, NamedArgsMixin, RandomSymbol
from sympy.core.sympify import _sympify

def rv(symbol, cls, args):
    dist = cls(sympify(args))
    pspace = MixturePSpace(symbol, dist)
    return pspace.value


def mixture_map(sym, distribution):
    x = Symbol('x')
    mixt_map = {
    'is_Continuous': lambda sym, distribution: SingleContinuousPSpace(sym,
        ContinuousDistributionHandmade(distribution.pdf, distribution.set)),
    'is_Joint' : lambda sym, distribution: JointPSpace(sym,
        JointDistributionHandmade(distribution.pdf, distribution.set)),
    'is_Discrete' : lambda sym, distribution: SingleDiscretePSpace(sym,
        DiscreteDistributionHandmade(distribution.pdf, distribution.set)),
    'is_Finite': lambda sym, distribution: SingleFinitePSpace(sym,
        FiniteDistributionHandmade(distribution.pdf, distribution.set))
    }
    return mixt_map[dist_type_check(distribution)](sym, distribution)

def dist_type_check(dist):
    dist = list(dist.wt_dict)[0]
    if isinstance(dist.pspace.distribution, ContinuousDistribution):
        return 'is_Continuous'
    elif isinstance(dist.pspace.distribution, JointDistribution):
        return 'is_Joint'
    elif isinstance(dist.pspace.distribution, DiscreteDistribution):
        return 'is_Discrete'
    else:
        return 'is_Finite'

class MixturePSpace(PSpace):
    """This class is just a temporary Probability Space as it will
    map to the original ProbabilitySpace based of the type of distribution
    passed
    """
    def __new__(cls, sym, distribution):
        if isinstance(sym, str):
            sym = Symbol(sym)
        if not isinstance(sym, Symbol):
            raise TypeError("sym should have been string or Symbol")
        pspace = mixture_map(sym, distribution)
        return Basic.__new__(cls, sym, distribution, pspace)

    @property
    def value(self):
        return self.args[2].value

class MixtureDistribution(Basic, NamedArgsMixin):
    """Represents the Mixture distribution"""
    _argnames = ('wt_dict')

    def __new__(cls, wt_dict):
        wt_list = [_sympify(wt) for wt in wt_dict.values()]
        set_ = list(wt_dict)[0].pspace.domain.set
        for rv in wt_dict.keys():
            if not isinstance(rv, RandomSymbol):
                raise TypeError("Each of element should be a random variable")
            if rv.pspace.domain.set != set_:
                raise ValueError("Each random variable should be defined on same set")
        for wt in wt_list:
            if not wt.is_positive:
                raise ValueError("Weight of each random variable should be positive")
        return Basic.__new__(cls, wt_dict)

    @property
    def set(self):
        return list(self.wt_dict)[0].pspace.distribution.set

    def pdf(self, x):
        y = Symbol('y')
        sum_wt = sum(self.wt_dict.values())
        wt_dict = {}
        for rv, wt in self.wt_dict.items():
            wt_dict[rv] = _sympify(self.wt_dict[rv]/sum_wt)
        pdf_ = S(0)
        if isinstance(list(wt_dict)[0].pspace.distribution, SingleFiniteDistribution):
            for rv in wt_dict.keys():
                pdf_ = pdf_ + wt_dict[rv]*rv.pspace.distribution.pmf(y)
        else:
            for rv in wt_dict.keys():
                pdf_ = pdf_ + wt_dict[rv]*rv.pspace.distribution.pdf(y)
        return Lambda(y, pdf_)(x)

def Mixture(name, wt_dict):
    """Creates a random variable with mixture distribution"""
    return rv(name, MixtureDistribution, wt_dict)
