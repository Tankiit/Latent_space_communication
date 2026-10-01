"""Conservative coordinate outer set of a bounded categorical-kernel MMD ball.

k(i,j)=1{i=j}. The MMD is Euclidean distance between probability vectors.
Truth in the Euclidean ball implies truth in the coordinate outer set used by LPs.
Reference uncertainty adds a maximum per-source embedding radius, by triangle inequality.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


def embedding_radius(n, alpha=.05):
    if n < 1 or not 0 < alpha < 1:
        raise ValueError('n >= 1 and 0 < alpha < 1 required')
    return float((1 + np.sqrt(2*np.log(1/alpha))) / np.sqrt(n))

def weight_interval(ball, index):
    """Extrema of one mixture weight over the feasible set; the plan's feasible-weight diameter.

    This is a marginal weight range, NOT a per-cell conditional. It is the quantity whose
    contraction the prevalence claim must bound, so it is reported separately from the widths.
    """
    a, b = ball.inequalities()
    k = len(ball.conditionals)
    if not 0 <= int(index) < k:
        raise ValueError('Weight index out of range')
    cost = np.zeros(k); cost[int(index)] = 1.
    out = {}
    for name, sign in (('lower', 1.), ('upper', -1.)):
        result = linprog(sign*cost, A_ub=a, b_ub=b, A_eq=np.ones((1, k)), b_eq=[1],
                         bounds=(0, 1), method='highs')
        if not result.success:
            return dict(lower=np.nan, upper=np.nan, width=np.nan, status='empty_set')
        out[name] = float(np.clip(sign*result.fun, 0, 1))
    return dict(**out, width=out['upper']-out['lower'], status='optimal')


@dataclass
class MMDBall:
    conditionals: np.ndarray
    empirical: np.ndarray
    eps_n: float
    reference_radius: float = 0.
    def __post_init__(self):
        self.conditionals = np.asarray(self.conditionals, dtype=float)
        self.empirical = np.asarray(self.empirical, dtype=float)
        if self.conditionals.ndim != 2 or self.empirical.shape != (self.conditionals.shape[1],):
            raise ValueError('Incompatible dimensions')
        if (not np.all(np.isfinite(self.conditionals)) or np.any(self.conditionals < 0)
            or not np.allclose(self.conditionals.sum(1), 1)
            or np.any(self.empirical < 0) or not np.isclose(self.empirical.sum(),1)
            or not np.isfinite(self.eps_n) or self.eps_n < 0 or self.reference_radius < 0):
            raise ValueError('Invalid probability inputs or radius')
    @classmethod
    def from_log(cls, conditionals, log, alpha=.05, reference_radius=0.):
        log = np.asarray(log)
        d = np.asarray(conditionals).shape[1]
        if log.ndim != 1 or not len(log) or not np.issubdtype(log.dtype,np.integer) or np.any((log<0)|(log>=d)):
            raise ValueError('Log must contain valid integer cell IDs')
        empirical = np.bincount(log,minlength=d) / len(log)
        return cls(conditionals, empirical, embedding_radius(len(log),alpha), reference_radius)
    @property
    def radius(self):
        return self.eps_n + self.reference_radius
    def inequalities(self):
        a = self.conditionals.T
        return np.vstack([a,-a]), np.concatenate([self.empirical+self.radius,-self.empirical+self.radius])
    def feasible(self):
        a,b = self.inequalities()
        result = linprog(np.zeros(len(self.conditionals)), A_ub=a, b_ub=b,
                         A_eq=np.ones((1,len(self.conditionals))), b_eq=[1], bounds=(0,None),method='highs')
        return result.success
    def covers_truth(self, marginal):
        return bool(np.linalg.norm(np.asarray(marginal)-self.empirical) <= self.eps_n + 1e-12)
    def outer_covers(self, weights):
        return bool(np.max(np.abs(np.asarray(weights)@self.conditionals-self.empirical)) <= self.radius + 1e-12)
    def resolution(self, w_from, w_to):
        return float(np.linalg.norm((np.asarray(w_to)-np.asarray(w_from))@self.conditionals)/self.radius)
    def ratio_interval(self, numerator_lo, numerator_hi, denominator, zero_tol=1e-10):
        """Linear-fractional extrema via Charnes-Cooper LP, not finite-grid enumeration.
        If any feasible model gives zero probe mass, return a vacuous policy interval.
        Conditional probabilities at zero-mass inputs are not identified by the joint law.
        """
        a,b = self.inequalities()
        k = len(self.conditionals)
        den = np.asarray(denominator,dtype=float)
        if den.shape != (k,) or np.any(den<0):
            raise ValueError('Invalid denominator')
        mass = linprog(den,A_ub=a,b_ub=b,A_eq=np.ones((1,k)),b_eq=[1],bounds=(0,None),method='highs')
        if not mass.success:
            return dict(lower=np.nan,upper=np.nan,status='empty_set')
        if mass.fun <= zero_tol:
            return dict(lower=0.,upper=1.,status='zero_mass_policy')
        # u=t*w, t=1/(den.w); a.u <= b*t; sum(u)=t; den.u=1.
        aub = np.column_stack([a,-b])
        aeq = np.vstack([np.r_[np.ones(k),-1.],np.r_[den,0.]])
        values=[]
        for num,sign in [(numerator_lo,1.),(numerator_hi,-1.)]:
            num=np.asarray(num,dtype=float)
            if num.shape != (k,) or np.any(num < -1e-12) or np.any(num > den+1e-12):
                raise ValueError('Numerator must lie between zero and denominator')
            result=linprog(np.r_[sign*num,0.],A_ub=aub,b_ub=np.zeros(len(b)),A_eq=aeq,
                           b_eq=[0,1],bounds=(0,None),method='highs')
            if not result.success:
                return dict(lower=np.nan,upper=np.nan,status='solver_failure')
            residual=max(np.max(aub@result.x),np.max(np.abs(aeq@result.x-[0,1])))
            if residual > 1e-7:
                return dict(lower=np.nan,upper=np.nan,status='solver_residual')
            values.append(float(np.clip(sign*result.fun,0,1)))
        return dict(lower=values[0],upper=values[1],status='optimal')
