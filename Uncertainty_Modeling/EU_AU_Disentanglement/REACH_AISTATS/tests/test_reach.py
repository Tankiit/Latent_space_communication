import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from data.worlds import make_world,make_separation_world,separation_gap,tilted_conditionals
from data.acquisitions import draw_log,draw_labelled_set
from models.ball import MMDBall,embedding_radius
from models.widths import label_intervals,credal_widths
from models.baselines import em_prevalence,feature_match
from test.inference import paired_contrast,seed_bootstrap_ci,equivalence_flat
from test.precision import contrast_precision
from train.cache import run_cache
from train.conditionals import condition_rows,summarise as conditional_summary
from train.matched import select_pair
from train.separation import separation_rows,summarise as separation_summary

class TestReach(unittest.TestCase):
    def test_exact_identified_weights(self):
        world=make_world(); ball=MMDBall(world.conditionals,world.marginal,0.)
        interval=ball.ratio_interval(world.conditionals[:,0]*world.theta,
                                     world.conditionals[:,0]*world.theta,world.conditionals[:,0])
        self.assertEqual(interval['status'],'optimal')
        self.assertAlmostEqual(interval['lower'],world.target(0),places=7)
        self.assertAlmostEqual(interval['upper'],world.target(0),places=7)
    def test_confounded_novel_mass_persists(self):
        world=make_world('confounded');ball=MMDBall(world.conditionals,world.marginal,0.)
        result=credal_widths(ball,0,world.theta[:2],world.theta[:2])
        self.assertGreater(result['W_unk'],.1)
        self.assertLessEqual(result['total_lower'],world.target(0)+1e-8)
        self.assertGreaterEqual(result['total_upper'],world.target(0)-1e-8)
    def test_zero_support_policy(self):
        world=make_world();ball=MMDBall(world.conditionals,world.marginal,0.)
        result=credal_widths(ball,3,world.theta[:2],world.theta[:2])
        self.assertEqual(result['total_status'],'zero_mass_policy')
        self.assertEqual(result['W_total'],1.)
        self.assertTrue(np.isnan(world.target(3)))
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): embedding_radius(0)
        with self.assertRaises(ValueError): draw_labelled_set(make_world(),10,.5,0)
        with self.assertRaises(ValueError): MMDBall.from_log(make_world().conditionals,np.array([-1]),.05)
    def test_empty_set(self):
        world=make_world();ball=MMDBall(world.conditionals,np.array([0,0,0,1.]),0.)
        self.assertFalse(ball.feasible())
        self.assertEqual(credal_widths(ball,0,world.theta[:2],world.theta[:2])['total_status'],'empty_set')
    def test_labels_ignore_log(self):
        world=make_world();lo,hi=label_intervals(draw_labelled_set(world,256,.2,2),.2)
        outputs=[]
        for n in [64,1024]:
            ball=MMDBall.from_log(world.conditionals,draw_log(world,n,2))
            outputs.append(credal_widths(ball,0,lo,hi)['W_label'])
        self.assertEqual(outputs[0],outputs[1])
    def test_matched_contrast_holds_nuisances(self):
        frame=pd.DataFrame([dict(world_id='w',probe_id=0,seed=0,n=n,m=m,eta=0,W_cls=n+m)
                            for n in [10,20] for m in [2,4]])
        result=paired_contrast(frame,'W_cls','n',10,20)
        self.assertEqual(len(result),2)
        np.testing.assert_array_equal(result.difference,[10,10])
    def test_bootstrap_groups(self):
        result=seed_bootstrap_ci([1,1,-1,-1],[0,0,1,1],100)
        self.assertEqual(result['independent_replicates'],2)
        self.assertEqual(result['estimate'],0)
        self.assertFalse(equivalence_flat(result,.01))
    def test_baseline_misspecification(self):
        world=make_world()
        self.assertIsNone(em_prevalence(world.conditionals[:2],np.array([2])))
        fit=feature_match(world.conditionals,world.marginal)
        self.assertLess(fit['residual'],1e-6)
    def test_joint_truth_inclusion(self):
        world=make_world();count=0
        for seed in range(30):
            ball=MMDBall.from_log(world.conditionals,draw_log(world,256,seed),.025)
            lo,hi=label_intervals(draw_labelled_set(world,256,.2,seed),.2,.025)
            for cell in [0,1,2]:
                result=credal_widths(ball,cell,lo,hi)
                count+=result['total_lower']-1e-8<=world.target(cell)<=result['total_upper']+1e-8
        self.assertGreaterEqual(count,85)  # numerical integration check, not proof of coverage
    def test_cached_feature_adapter(self):
        rng=np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=np.repeat([0,1],100)
            ref=rng.normal(size=(200,2))+source[:,None]*3
            np.savez(root/'cache.npz',X_partition=rng.normal(size=(100,2)),X_reference=ref,
                     source_reference=source,target_reference=(rng.random(200)<np.where(source==0,.2,.8)).astype(int),
                     X_log=ref[:50],X_eval=ref[50:60],target_eval=np.zeros(10,dtype=int))
            run_cache(root/'cache.npz',root/'out',cells=3)
            self.assertTrue((root/'out/rows.csv').exists())

class TestFailureWorldsAndPlanning(unittest.TestCase):
    """Experiments 3b, 5 and 6, plus replication planning. Grids are deliberately tiny:
    these check the machinery and the direction of each claim, not the reported evidence."""
    def test_separation_gap_matches_construction(self):
        for sep in (1.,.25):
            self.assertAlmostEqual(separation_gap(make_separation_world(sep)),.4*sep*np.sqrt(2))
        with self.assertRaises(ValueError): make_separation_world(0.)
    def test_small_gap_resolves_slower_without_losing_coverage(self):
        summary=separation_summary(separation_rows([1.,.1],[512],range(6)),.2)
        self.assertEqual(summary['best_case']['sep'],1.)
        self.assertEqual(summary['worst_case']['sep'],.1)
        # A near-singular embedding must NOT report the narrower interval.
        self.assertGreater(summary['worst_case']['W_cls_at_max_n'],
                           summary['best_case']['W_cls_at_max_n'])
        self.assertTrue(summary['validity_preserved_at_all_separations'])
    def test_tilt_leaves_declared_family(self):
        world=make_world();tilted=tilted_conditionals(world,.5)
        np.testing.assert_allclose(tilted[:2,-1],.5*world.conditionals[:2,0])
        moved=float(world.weights[:2]@tilted[:2,-1])
        residual=feature_match(world.conditionals,world.weights@tilted)['residual']
        self.assertGreaterEqual(residual,moved-1e-9)  # unsupported cell bounds the residual below
        with self.assertRaises(ValueError): tilted_conditionals(world,1.5)
    def test_dropping_reference_radius_loses_coverage(self):
        world=make_world()
        frame=pd.concat([condition_rows(world,condition,[32],[512],range(6),m=64)
                         for condition in ('positive_control','ignored_reference')],ignore_index=True)
        table={row['condition']:row for row in conditional_summary(frame)['table']}
        self.assertGreater(table['positive_control']['mean_radius'],
                           table['ignored_reference']['mean_radius'])
        self.assertGreaterEqual(table['positive_control']['weight_coverage_rate'],
                                table['ignored_reference']['weight_coverage_rate'])
    def test_shifted_deployment_is_flagged_incompatible(self):
        world=make_world()
        frame=condition_rows(world,'conditional_shift',[2048],[2048],range(4),shift=.5,m=64)
        self.assertTrue(frame.incompatible.all())
        self.assertTrue((frame.declared_family_residual>frame.radius).all())
    def test_matched_selection_respects_band_and_ceiling(self):
        arms=[(.5,.3,.05),(.505,.05,.3),(.95,.9,0.)]  # third arm sits above the vacuous ceiling
        frame=pd.DataFrame([dict(arm_id=i,arm=f'a{i}',seed=seed,baseline=base,
                                 gain_log=gl,gain_label=gb)
                            for i,(base,gl,gb) in enumerate(arms) for seed in range(3)])
        pair=select_pair(frame,band=.02,max_baseline=.92)
        self.assertEqual((pair['log_favoured_arm'],pair['label_favoured_arm']),(0,1))
        self.assertIsNone(select_pair(frame,band=1e-6,max_baseline=.92))
    def test_precision_planning_scales_with_target(self):
        frame=pd.DataFrame([dict(world_id='w',probe_id=0,region='on_support',seed=seed,n=n,m=4,
                                 eta=0.,W_cls=.1*n+(.05*seed if n==20 else 0.))
                            for n in (10,20) for seed in range(5)])
        predicted={('W_cls','n','w'):1}
        loose=contrast_precision(frame,predicted,.5,n_boot=200)[0]
        tight=contrast_precision(frame,predicted,.01,n_boot=200)[0]
        self.assertEqual(loose['independent_seeds'],5)
        self.assertGreater(tight['seeds_for_target'],loose['seeds_for_target'])
        self.assertTrue(loose['target_met'])
        self.assertFalse(tight['target_met'])

if __name__=='__main__': unittest.main()
