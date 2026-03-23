"""Adversarial tests — Stress-test the quant system under worst-case conditions.

These tests intentionally break assumptions, perturb inputs, and inject
noise to verify the system degrades gracefully rather than catastrophically.
"""
from tests.adversarial.probability_perturbation import ProbabilityPerturbationTest
from tests.adversarial.best_bet_removal import BestBetRemovalTest
from tests.adversarial.noise_injection import NoiseInjectionTest
from tests.adversarial.assumption_distortion import AssumptionDistortionTest
from tests.adversarial.runner import AdversarialTestRunner

__all__ = [
    "ProbabilityPerturbationTest",
    "BestBetRemovalTest",
    "NoiseInjectionTest",
    "AssumptionDistortionTest",
    "AdversarialTestRunner",
]
