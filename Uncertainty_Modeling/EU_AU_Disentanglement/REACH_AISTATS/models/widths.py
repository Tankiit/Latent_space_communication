"""Explicit distinct estimands: these three widths are NOT additive components.

W_cls: P(Z=source 1 | X=cell, Z is known), over feasible source weights.
W_label: P(Y=1 | X=cell, Z is known), at fixed reference weights, over label intervals.
W_unk: P(Z=novel | X=cell), over feasible source weights.
W_total: P(Y=1 | X=cell), over weights, known-source label intervals, and a free novel label.
"""
import numpy as np


def label_intervals(labels, eta, alpha=.05):
    labels=np.asarray(labels,dtype=float)
    if labels.ndim != 2 or labels.shape[0] != 2 or labels.shape[1]<1 or np.any((labels!=0)&(labels!=1)):
        raise ValueError('Expected two strata of binary noisy labels')
    if not 0 <= eta < .5 or not 0 < alpha < 1:
        raise ValueError('Invalid eta or confidence level')
    # Union bound over known sources, plus two-sided Hoeffding within each.
    r=np.sqrt(np.log(2*len(labels)/alpha)/(2*labels.shape[1]))
    means=labels.mean(1)
    lo=np.clip((means-r-eta)/(1-2*eta),0,1)
    hi=np.clip((means+r-eta)/(1-2*eta),0,1)
    return lo,hi


def credal_widths(ball, cell, label_lo, label_hi, reference_weights=(.5,.5)):
    d=ball.conditionals[:,cell]
    known_den=d.copy(); known_den[-1]=0
    cls_num=np.zeros_like(d); cls_num[0]=d[0]
    cls=ball.ratio_interval(cls_num,cls_num,known_den)
    unk_num=np.zeros_like(d); unk_num[-1]=d[-1]
    unk=ball.ratio_interval(unk_num,unk_num,d)
    total=ball.ratio_interval(d*np.r_[label_lo,0.],d*np.r_[label_hi,1.],d)
    anchor=np.asarray(reference_weights)*d[:2]
    if anchor.sum() > 0:
        anchor/=anchor.sum()
        lab=dict(lower=float(anchor@label_lo),upper=float(anchor@label_hi),status='fixed_anchor')
    else:
        lab=dict(lower=np.nan,upper=np.nan,status='undefined_known_probe')
    out={}
    for name,interval in [('cls',cls),('label',lab),('unk',unk),('total',total)]:
        out.update({f'{name}_lower':interval['lower'],f'{name}_upper':interval['upper'],
                    f'W_{name}':interval['upper']-interval['lower'],f'{name}_status':interval['status']})
    return out
