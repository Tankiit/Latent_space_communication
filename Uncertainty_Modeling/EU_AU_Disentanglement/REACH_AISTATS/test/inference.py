"""Paired contrasts with all other intervention factors held fixed."""
import numpy as np
import pandas as pd
from scipy.stats import beta


def paired_contrast(df,width,knob,lo,hi,by=None):
    if by is None:
        by=[c for c in ['world_id','probe_id','seed','n','m','eta'] if c!=knob]
    a=df.loc[df[knob]==lo,by+[width]].rename(columns={width:'low'})
    b=df.loc[df[knob]==hi,by+[width]].rename(columns={width:'high'})
    paired=a.merge(b,on=by,validate='one_to_one')
    paired['difference']=paired.high-paired.low
    return paired


def seed_bootstrap_ci(diffs,group,n_boot=2000,level=.95,seed=0):
    frame=pd.DataFrame({'difference':diffs,'group':group}).dropna()
    # Equal-weight independent replicate means; preserve probes/conditions inside a replicate.
    means=frame.groupby('group').difference.mean().to_numpy()
    if len(means)<2:
        return dict(estimate=float(means.mean()) if len(means) else np.nan,lower=np.nan,upper=np.nan,
                    independent_replicates=len(means),status='insufficient_replicates')
    rng=np.random.default_rng(seed)
    boot=rng.choice(means,(n_boot,len(means)),replace=True).mean(1)
    lo,hi=np.quantile(boot,[(1-level)/2,(1+level)/2])
    return dict(estimate=float(means.mean()),lower=float(lo),upper=float(hi),
                independent_replicates=len(means),status='ok')


def equivalence_flat(diffs_ci,band):
    return bool(np.isfinite(diffs_ci['lower']) and diffs_ci['lower']>=-band and diffs_ci['upper']<=band)


def coverage_at_least_nominal(covered,nominal=.95,level=.95):
    obs=np.asarray(covered,dtype=bool); n=len(obs); s=int(obs.sum())
    if not n:
        raise ValueError('No independent coverage replicates')
    lower=0. if s==0 else float(beta.ppf(1-level,s,n-s+1))
    return dict(rate=s/n,lower=lower,independent_replicates=n,nominal=nominal,
                demonstrates_at_least_nominal=lower>=nominal)
