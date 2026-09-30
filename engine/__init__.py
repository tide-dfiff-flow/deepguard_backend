"""Engine: matchup orchestration, fidelity judge, reporting."""
from .sample import Sample, load_protocol, split_by_label  # noqa: F401
from .fidelity_judge import FidelityJudge, precheck  # noqa: F401
from .matchup import MatchupEngine  # noqa: F401
from .report import generate_reports  # noqa: F401
