"""Metric computation for the DeepGuard-Audio backend."""
from .detection import (  # noqa: F401
    compute_eer, eer_from_spoof_prob, auc_from_spoof_prob,
    fpr_fnr_at_threshold, compute_tdcf, min_dcf, all_detection_metrics,
)
from .fidelity import (  # noqa: F401
    si_sdr, stoi, pesq_score, fidelity_bundle,
)
from .attack_metrics import (  # noqa: F401
    attack_success_rate, confidence_suppression, attack_metric_bundle,
)
from .content import cer, wer, content_preservation  # noqa: F401
from .scoring import (  # noqa: F401
    quality_coefficient, core_asr, attack_score, defense_score, efficiency,
)
