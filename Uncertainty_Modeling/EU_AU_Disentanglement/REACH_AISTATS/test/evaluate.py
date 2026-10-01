"""Evaluate fitted intervals; evaluation labels never enter fitting."""
import numpy as np

def evaluate_intervals(frame, target_eval=None):
    frame['prediction']=np.where(frame.total_lower>.5,1,np.where(frame.total_upper<.5,0,-1))
    # These are empirical selective-decision metrics, NOT pointwise posterior coverage.
    metrics=dict(acceptance=float((frame.prediction>=0).mean()),posterior_coverage='unobserved',
             mean_total_width=float(frame.W_total.mean()))
    if target_eval is not None:
        y=np.asarray(target_eval)
        if y.shape!=(len(frame),) or not np.all(np.isin(y,[0,1])):
            raise ValueError('Invalid evaluation labels')
        accepted=frame.prediction.to_numpy()>=0
        metrics['accepted_error']=float(np.mean(frame.prediction.to_numpy()[accepted]!=y[accepted])) if accepted.any() else None
        frame['target_eval']=y
    return frame, metrics
