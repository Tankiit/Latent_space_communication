"""
Implementation of the Five-Step Hurdle Audit Battery.
See [[04-hurdle-test-and-audit]] for specification.
"""

import math
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from .interventions import evaluate_leakage


class HurdleAuditor:
    """
    Executes the 5-step Hurdle Audit Battery on candidate concepts and trained models:
    1. Structural Zero Test (Vuong Test: Hurdle vs Tobit/One-Part)
    2. Extent Decoupling Test (Non-trivial conditional variance)
    3. Absence Invariance Audit (Counterfactual Leakage Delta_leak = 0)
    4. Conditional Calibration Audit (Gate ECE & Extent Coverage)
    """

    @staticmethod
    def test_structural_zero(
        values: np.ndarray,
        eps: float = 1e-4,
        log_scale: bool = True,
    ) -> Dict[str, Any]:
        """
        Test 1: Evaluates whether zeros form a structural hurdle using Vuong test
        comparing a Two-Part Hurdle vs a One-Part censored Tobit model.

        Args:
            values: Array of candidate concept values with point mass at 0.
            eps: Numerical threshold distinguishing zeros from positive extent.
            log_scale: Whether positive extents follow LogNormal (standard for physical measurements).
        """
        zeros_mask = (values <= eps)
        prop_zeros = np.mean(zeros_mask)
        n = len(values)

        if prop_zeros < 0.05 or prop_zeros > 0.95:
            return {
                "passed": False,
                "reason": f"Proportion of zeros {prop_zeros:.3f} outside admissible hurdle range [0.05, 0.95].",
                "prop_zeros": float(prop_zeros),
                "vuong_stat": 0.0,
                "p_value": 1.0,
            }

        pos_values = values[~zeros_mask]
        n_pos = len(pos_values)
        n_zero = np.sum(zeros_mask)

        # 1. Fit Two-Part Hurdle
        pi_hat = n_pos / n
        log_lik_hurdle = np.zeros(n)
        log_lik_hurdle[zeros_mask] = np.log(1.0 - pi_hat)

        if log_scale:
            trans_pos = np.log(pos_values)
            mu_pos = np.mean(trans_pos)
            sigma_pos = np.std(trans_pos) + 1e-6
            log_lik_hurdle[~zeros_mask] = (
                np.log(pi_hat)
                - 0.5 * np.log(2 * np.pi)
                - np.log(sigma_pos)
                - trans_pos
                - 0.5 * ((trans_pos - mu_pos) / sigma_pos) ** 2
            )
        else:
            mu_pos = np.mean(pos_values)
            sigma_pos = np.std(pos_values) + 1e-6
            log_lik_hurdle[~zeros_mask] = (
                np.log(pi_hat)
                - 0.5 * np.log(2 * np.pi)
                - np.log(sigma_pos)
                - 0.5 * ((pos_values - mu_pos) / sigma_pos) ** 2
            )

        # 2. Fit Censored Tobit (Left-censored at threshold c)
        try:
            from scipy.optimize import minimize
            from scipy.special import log_ndtr

            c = np.log(eps) if log_scale else eps
            trans_all = np.log(pos_values) if log_scale else pos_values

            def neg_tobit_ll(params):
                mu, log_sig = params
                sig = np.exp(log_sig) + 1e-6
                ll_0 = n_zero * log_ndtr((c - mu) / sig)
                z = (trans_all - mu) / sig
                ll_pos = np.sum(-0.5 * np.log(2 * np.pi) - log_sig - 0.5 * z ** 2)
                if log_scale:
                    ll_pos -= np.sum(trans_all)
                return -(ll_0 + ll_pos)

            opt_res = minimize(neg_tobit_ll, [0.0, 0.0], method="L-BFGS-B")
            mu_t, sig_t = opt_res.x[0], np.exp(opt_res.x[1]) + 1e-6

            log_lik_tobit = np.zeros(n)
            log_lik_tobit[zeros_mask] = log_ndtr((c - mu_t) / sig_t)
            z_pos = (trans_all - mu_t) / sig_t
            log_lik_tobit[~zeros_mask] = -0.5 * np.log(2 * np.pi) - np.log(sig_t) - 0.5 * z_pos ** 2
            if log_scale:
                log_lik_tobit[~zeros_mask] -= trans_all
        except Exception:
            log_lik_tobit = np.full(n, -2.0)

        # 3. Vuong Likelihood Ratio Test Statistic
        m_i = log_lik_hurdle - log_lik_tobit
        mean_m = np.mean(m_i)
        std_m = np.std(m_i) + 1e-8
        vuong_stat = (np.sqrt(n) * mean_m) / std_m

        passed = bool(vuong_stat > 1.96)

        return {
            "passed": passed,
            "prop_zeros": float(prop_zeros),
            "vuong_stat": float(vuong_stat),
            "p_value": float(0.5 * (1.0 - math.erf(vuong_stat / math.sqrt(2)))),
            "recommendation": "Use Hurdle" if passed else "Do not type as hurdle",
        }

    @staticmethod
    def test_extent_decoupling(
        states: np.ndarray,
        extents: np.ndarray,
        min_cv: float = 0.15,
        min_unique: int = 10,
    ) -> Dict[str, Any]:
        """
        Test 2: Verifies that extent has non-trivial continuous variance conditional on S=1.
        """
        active_mask = (states > 0.5)
        if not np.any(active_mask):
            return {"passed": False, "reason": "No active samples found."}

        active_extents = extents[active_mask]
        mean_e = np.mean(active_extents)
        std_e = np.std(active_extents)
        cv = std_e / (mean_e + 1e-8)
        num_unique = len(np.unique(np.round(active_extents, 4)))

        passed = bool(cv >= min_cv and num_unique >= min_unique)

        return {
            "passed": passed,
            "cv": float(cv),
            "num_unique": int(num_unique),
            "mean_extent": float(mean_e),
            "std_extent": float(std_e),
        }

    @staticmethod
    def test_absence_invariance(
        model: nn.Module,
        embeddings: torch.Tensor,
        concept_idx: int,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Test 3: Counterfactual Leakage Audit under do(S_j = 0).
        """
        leakage = evaluate_leakage(model, embeddings, concept_idx)
        passed = bool(leakage <= tolerance)

        return {
            "passed": passed,
            "delta_leak": float(leakage),
            "tolerance": float(tolerance),
            "status": "Invariance Preserved" if passed else "Absence Leakage Detected",
        }

    @staticmethod
    def test_conditional_calibration(
        pred_pi: np.ndarray,
        true_states: np.ndarray,
        pred_mu: np.ndarray,
        pred_sigma: np.ndarray,
        true_extents: np.ndarray,
        nominal_coverage: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Test 5: Checks calibration of the discrete gate and coverage of active extents.
        """
        # Gate ECE (10 bins)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        n = len(true_states)
        for i in range(10):
            mask = (pred_pi >= bins[i]) & (pred_pi < bins[i + 1])
            if np.any(mask):
                bin_acc = np.mean(true_states[mask])
                bin_conf = np.mean(pred_pi[mask])
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

        # Extent coverage on active subset
        active_mask = (true_states > 0.5)
        if np.any(active_mask):
            act_true_e = true_extents[active_mask]
            act_mu = pred_mu[active_mask]
            act_sigma = pred_sigma[active_mask]

            z_crit = 1.645  # 90% normal interval
            lower_bound = np.exp(act_mu - z_crit * act_sigma)
            upper_bound = np.exp(act_mu + z_crit * act_sigma)

            in_interval = (act_true_e >= lower_bound) & (act_true_e <= upper_bound)
            empirical_coverage = float(np.mean(in_interval))
        else:
            empirical_coverage = 0.0

        passed = (ece <= 0.08) and (abs(empirical_coverage - nominal_coverage) <= 0.06)

        return {
            "passed": bool(passed),
            "gate_ece": float(ece),
            "extent_coverage_90": float(empirical_coverage),
            "nominal_coverage": float(nominal_coverage),
        }
