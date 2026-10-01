"""Runnable pilot. Output is evidence about the declared finite-cell prototype only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from data.worlds import make_world
from data.acquisitions import draw_log,draw_labelled_set
from models.ball import MMDBall
from models.widths import label_intervals,credal_widths
from models.baselines import bootstrap_em,feature_match
import test.inference as inf
import train.predictions as P


def response_matrix(world,n_grid,m_grid,eta_grid,seeds,alpha=.05):
    rows=[]
    for seed in seeds:
        log=draw_log(world,max(n_grid),seed)
        for eta in eta_grid:
            labels=draw_labelled_set(world,max(m_grid),eta,seed)
            for n in n_grid:
                ball=MMDBall.from_log(world.conditionals,log[:n],alpha/2)
                for m in m_grid:
                    lo,hi=label_intervals(labels[:,:m],eta,alpha/2)
                    for cell,region in enumerate(world.regions):
                        row=dict(world_id=world.name,probe_id=cell,region=region,seed=seed,n=n,m=m,eta=eta,
                                 delta=float(world.weights[-1]),eps_n=ball.eps_n,alpha=alpha,
                                 ball_covers_truth=ball.covers_truth(world.marginal),
                                 outer_covers_truth=ball.outer_covers(world.weights),
                                 labels_cover_truth=bool(np.all(lo<=world.theta[:2]) and np.all(world.theta[:2]<=hi)),
                                 ball_resolution=ball.resolution([.5,.5,0],[.9,.1,0]),
                                 true_target=world.target(cell),true_probe_mass=float(world.marginal[cell]))
                        row.update(credal_widths(ball,cell,lo,hi))
                        truth=row['true_target']
                        row['total_covers_truth']=bool(row['total_lower']-1e-9<=truth<=row['total_upper']+1e-9) if np.isfinite(truth) else None
                        rows.append(row)
    return pd.DataFrame(rows)


def analyse(df,n_boot=1000):
    contrasts=[]
    for (width,knob,world_id),sign in P.PREDICTED.items():
        sub=df[(df.world_id==world_id)&df.region.isin(['on_support','boundary'])]
        levels=sorted(sub[knob].unique())
        if len(levels)<2:
            continue
        pairs=inf.paired_contrast(sub,width,knob,levels[0],levels[-1])
        ci=inf.seed_bootstrap_ci(pairs.difference,pairs.seed,n_boot=n_boot)
        if sign is None:
            decision='exploratory'
        elif not np.isfinite(ci['lower']):
            decision='inconclusive'
        elif sign==0:
            decision='equivalent' if inf.equivalence_flat(ci,P.EQUIV_BAND) else 'not_established'
        elif (sign<0 and ci['upper'] < -P.MIN_EFFECT) or (sign>0 and ci['lower'] > P.MIN_EFFECT):
            decision='supported'
        elif (sign<0 and ci['lower'] > P.MIN_EFFECT) or (sign>0 and ci['upper'] < -P.MIN_EFFECT):
            decision='opposite_effect'
        else:
            decision='inconclusive'
        contrasts.append(dict(width=width,knob=knob,world_id=world_id,predicted_sign=sign,decision=decision,**ci))
    coverage=[]
    # Ball coverage is one event per world, n, seed; do not count duplicated probes/m/eta.
    unique=df.drop_duplicates(['world_id','n','seed'])
    for (world,n),sub in unique.groupby(['world_id','n']):
        coverage.append(dict(world_id=world,n=int(n),**inf.coverage_at_least_nominal(sub.ball_covers_truth,.975)))
    return contrasts,coverage


def plot_results(df,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    sub=df[(df.m==df.m.max())&(df.eta==df.eta.min())]
    for ax,width in zip(axes,['W_cls','W_label','W_unk']):
        for (world,region),group in sub.groupby(['world_id','region']):
            if region not in ['on_support','boundary','off_support']:
                continue
            values=group.groupby('n')[width].mean()
            ax.plot(values.index,values.values,marker='o',label=f'{world}/{region}')
        ax.set(xscale='log',xlabel='Unlabelled log size',ylabel=width,ylim=(-.02,1.02))
    axes[-1].legend(fontsize=6,loc='best')
    fig.suptitle('Finite-cell pilot: distinct estimands; no additive decomposition')
    fig.tight_layout(); fig.savefig(out/'widths.png',dpi=160); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/pilot')
    parser.add_argument('--seeds',type=int,default=8)
    parser.add_argument('--seed-start',type=int,default=0,
                        help='first seed; give shards disjoint ranges so replications stay independent')
    parser.add_argument('--n',type=int,nargs='+',default=[64,256,1024])
    parser.add_argument('--m',type=int,nargs='+',default=[32,128])
    parser.add_argument('--eta',type=float,nargs='+',default=[0,.2])
    parser.add_argument('--worlds',nargs='+',default=['separated','absent','confounded'])
    parser.add_argument('--bootstrap',type=int,default=50)
    parser.add_argument('--precision-target',type=float,default=.01,
                        help='target bootstrap half-width for replication planning')
    args=parser.parse_args(argv)
    if args.seeds<2 or min(args.n+args.m)<1 or args.bootstrap<1 or args.seed_start<0:
        parser.error('At least two seeds, a nonnegative seed start, and positive budgets/bootstrap required')
    seeds=range(args.seed_start,args.seed_start+args.seeds)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=vars(args),protocol=P.PROTOCOL,
                  predictions_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
                  analysis_kind='pilot_not_confirmatory',alpha_total=.05)
    project_root = Path(__file__).resolve().parents[1]
    manifest['source_sha256'] = {str(f.relative_to(project_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for folder in ('models', 'data', 'train', 'test')
                                 for f in sorted((project_root / folder).glob('*.py'))}
    manifest['source_sha256']['main.py'] = hashlib.sha256((project_root / 'main.py').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    frames=[]; baselines=[]
    for name in args.worlds:
        world=make_world(name)
        frames.append(response_matrix(world,args.n,args.m,args.eta,seeds))
        for seed in seeds:
            log=draw_log(world,max(args.n),seed)
            for n in args.n:
                # Do not offer an oracle novel conditional to the closed-set baseline silently.
                for kind,a in [('closed_set',world.conditionals[:2]),('oracle_augmented',world.conditionals)]:
                    result=bootstrap_em(a,log[:n],args.bootstrap,seed=seed+333)
                    baselines.append(dict(world_id=name,seed=seed,n=n,baseline=kind,**result))
                    fit=feature_match(a,np.bincount(log[:n],minlength=a.shape[1])/n)
                    baselines.append(dict(world_id=name,seed=seed,n=n,baseline='simple_feature_match_'+kind,**fit))
    df=pd.concat(frames,ignore_index=True); df.to_csv(out/'rows.csv',index=False)
    contrasts,coverage=analyse(df)
    (out/'contrasts.json').write_text(json.dumps(contrasts,indent=2,allow_nan=False))
    (out/'coverage.json').write_text(json.dumps(coverage,indent=2))
    import test.precision as prec
    (out/'precision.json').write_text(json.dumps(
        prec.summarise(prec.contrast_precision(df,P.PREDICTED,args.precision_target),
                       args.precision_target),indent=2))
    (out/'em_baselines.json').write_text(json.dumps(baselines,indent=2))
    plot_results(df,out)
    print(f'Saved {len(df)} probe rows; {len(contrasts)} paired acquisition contrasts to {out}')
    print('Pilot only; off-support uses a vacuous policy, and widths are not additive.')

if __name__=='__main__':
    main()
