"""EMQ on oracle/estimated discrete conditionals; known-only and augmented fits explicit."""
import numpy as np

def em_prevalence(conditionals, log, max_iter=500, tol=1e-9):
    a=np.asarray(conditionals,dtype=float)
    counts=np.bincount(log,minlength=a.shape[1]).astype(float)
    if np.any((counts>0)&(a.sum(0)==0)):
        return None  # closed-set misspecification; do not silently discard novel observations
    w=np.ones(len(a))/len(a)
    for _ in range(max_iter):
        den=w@a
        resp=np.divide(w[:,None]*a,den[None,:],out=np.zeros_like(a),where=den[None,:]>0)
        new=(resp@counts)/counts.sum()
        if np.max(np.abs(new-w)) < tol:
            return new
        w=new
    return w

def bootstrap_em(conditionals, log, n_boot=50, alpha=.05, seed=0):
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(n_boot):
        w=em_prevalence(conditionals,rng.choice(log,len(log),replace=True))
        if w is not None:
            samples.append(w)
    if not samples:
        return dict(status='model_misspecified',lower=None,upper=None)
    samples=np.asarray(samples)
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2],axis=0)
    probe_intervals={}
    # Match REACH's W_cls estimand, preserving dependence between bootstrap weights.
    for cell in range(np.asarray(conditionals).shape[1]):
        known_mass=samples[:,:2]@np.asarray(conditionals)[:2,cell]
        valid=known_mass>1e-12
        if not valid.all():
            probe_intervals[str(cell)]=dict(lower=None,upper=None,status='undefined_known_probe')
        else:
            values=samples[:,0]*np.asarray(conditionals)[0,cell]/known_mass
            left,right=np.quantile(values,[alpha/2,1-alpha/2])
            probe_intervals[str(cell)]=dict(lower=float(left),upper=float(right),status='bootstrap')
    return dict(status='percentile_bootstrap_no_finite_sample_guarantee',lower=lo.tolist(),upper=hi.tolist(),
                known_posterior_intervals=probe_intervals)


def feature_match(conditionals, empirical):
    """Simple simplex least-squares feature matching; no claim to reproduce full robust DFM."""
    from scipy.optimize import minimize
    a=np.asarray(conditionals,dtype=float)
    empirical=np.asarray(empirical,dtype=float)
    result=minimize(lambda w: np.sum((w@a-empirical)**2),np.ones(len(a))/len(a),
                    method='SLSQP',bounds=[(0,1)]*len(a),
                    constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],
                    options=dict(ftol=1e-12,maxiter=1000))
    if not result.success:
        raise RuntimeError('Feature matching solver failed: '+result.message)
    return dict(weights=result.x.tolist(),residual=float(np.linalg.norm(result.x@a-empirical)),
                status='point_estimate_no_confidence_interval')
