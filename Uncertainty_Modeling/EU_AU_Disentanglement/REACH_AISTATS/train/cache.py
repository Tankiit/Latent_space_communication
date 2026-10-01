"""Cached-feature adapter; guarantees concern fixed cells and an assumed mixture family.
Partition features must be an independent development pool. Evaluation labels are read
only after fitting. Two known source strata; the novel conditional is an explicit candidate
uniform over cells unless q_cells is supplied. This does not identify arbitrary unseen classes.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from models.ball import MMDBall,embedding_radius
from models.widths import label_intervals,credal_widths
from models.baselines import feature_match
from data.features import nearest_cells
from data.cache import load_cache
from test.evaluate import evaluate_intervals


def run_cache(path,out,cells=8,eta=0.,seed=0):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with load_cache(path) as data:
        required=['X_partition','X_reference','source_reference','target_reference','X_log','X_eval']
        missing=[key for key in required if key not in data]
        if missing:
            raise ValueError('Missing cache keys: '+', '.join(missing))
        dev=np.asarray(data['X_partition'],dtype=float)
        if dev.ndim!=2 or not np.all(np.isfinite(dev)) or len(dev)<cells or cells<2:
            raise ValueError('Independent partition pool too small or invalid')
        centres,_=kmeans2(dev,cells,minit='++',seed=seed,missing='raise')
        ref_cells=nearest_cells(data['X_reference'],centres)
        sources=np.asarray(data['source_reference'])
        targets=np.asarray(data['target_reference'])
        if sources.shape!=(len(ref_cells),) or targets.shape!=sources.shape or not np.all(np.isin(sources,[0,1])):
            raise ValueError('Reference sources must be 0 or 1 and row aligned')
        if not np.all(np.isin(targets,[0,1])):
            raise ValueError('Reference targets must be binary')
        pools=[np.flatnonzero(sources==k) for k in range(2)]
        counts=[len(pool) for pool in pools]
        if min(counts)<1:
            raise ValueError('Both known sources need labelled reference observations')
        a=np.array([np.bincount(ref_cells[pool],minlength=cells)/len(pool) for pool in pools])
        q=np.asarray(data['q_cells'],dtype=float) if 'q_cells' in data else np.ones(cells)/cells
        if q.shape!=(cells,) or np.any(q<0) or not np.isclose(q.sum(),1):
            raise ValueError('q_cells must be a declared candidate cell distribution')
        conditionals=np.vstack([a,q])
        log=nearest_cells(data['X_log'],centres)
        # Total confidence failure budget: log, references, label channel each alpha/3.
        alpha=.05
        ref_radius=max(embedding_radius(count,alpha/6) for count in counts)
        ball=MMDBall.from_log(conditionals,log,alpha/3,ref_radius)
        m=min(counts)
        # First m observations within each stratum; row order must not use target outcomes.
        labs=np.vstack([targets[pool[:m]] for pool in pools])
        lo,hi=label_intervals(labs,eta,alpha/3)
        eval_cells=nearest_cells(data['X_eval'],centres)
        cached={int(c):credal_widths(ball,int(c),lo,hi) for c in np.unique(eval_cells)}
        rows=[dict(probe_id=i,cell=int(c),**cached[int(c)]) for i,c in enumerate(eval_cells)]
        frame=pd.DataFrame(rows)
        if frame.empty:
            raise ValueError('Empty evaluation pool')
        frame, metrics = evaluate_intervals(frame, data['target_eval'] if 'target_eval' in data else None)
        fit=feature_match(a,np.bincount(log,minlength=cells)/len(log))
        metrics['known_mixture_diagnostic']=dict(**fit,
            incompatible_at_declared_radius=fit['residual']>ball.radius,
            interpretation='incompatibility with known mixture, not identification of conditional shift')
        frame.to_csv(out/'rows.csv',index=False)
        np.savez(out/'fitted.npz',centres=centres,conditionals=conditionals,label_lower=lo,label_upper=hi)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (out/'manifest.json').write_text(json.dumps(dict(seed=seed,cells=cells,eta_known=eta,n=len(log),
            reference_counts=counts,label_budget_per_source=m,alpha=alpha,reference_radius=ref_radius,
            eps_n=ball.eps_n,novel_family='fixed_q_cells_candidate',
            assumptions=['independent partition/reference/log/evaluation pools','iid within pools',
                'deployment is in declared mixture family','source-stratified reference acquisition',
                'Y independent of input cell conditional on source','known symmetric binary label-noise rate'],
            scope='finite-cell probabilities; no guarantee for individual original feature vectors'),indent=2))
    print('Saved finite-cell deployment audit to',out)

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('cache');parser.add_argument('--out',default='results/real')
    parser.add_argument('--cells',type=int,default=8);parser.add_argument('--eta',type=float,default=0.)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args(argv);run_cache(args.cache,args.out,args.cells,args.eta,args.seed)


if __name__ == "__main__":
    main()
