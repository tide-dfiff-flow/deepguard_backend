"""DeepGuard-Audio attack/defense adversarial evaluation backend.

A self-contained backend that, for any combination of attack methods and
defense methods, automatically computes the corresponding attack metrics
(ASR, confidence suppression, fidelity) and defense metrics (EER, AUC,
minDCF, per-attack EER) following the competition specification.

Core dependencies: numpy, scipy, scikit-learn, torch (CPU).
Optional dependencies (auto-detected, graceful degradation):
soundfile, librosa, pystoi, pesq, matplotlib.
"""
__version__ = "0.1.0"
