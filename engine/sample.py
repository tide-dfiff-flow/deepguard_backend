"""Sample dataclass and protocol loading (spec §3.2)."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..audio_io import load_audio, TARGET_SR

LABEL_BONAFIDE = 0
LABEL_SPOOF = 1


@dataclass
class Sample:
    sample_id: str
    audio: np.ndarray
    sample_rate: int = TARGET_SR
    label: int = -1            # 0=bonafide, 1=spoof; -1 = unknown (defense side)
    attack_id: str = "unknown"
    speaker_id: str = "unknown"
    domain: str = "unknown"
    audio_path: Optional[str] = None

    def as_attack_input(self) -> dict:
        """What the attack receives (spec §3.3): no label/generator/etc."""
        return {"sample_id": self.sample_id,
                "audio": self.audio.astype(np.float32, copy=False),
                "sample_rate": self.sample_rate}

    def as_defense_input(self) -> dict:
        """What the defense receives (spec §3.4): no label/generator/attack."""
        return {"sample_id": self.sample_id,
                "audio": self.audio.astype(np.float32, copy=False),
                "sample_rate": self.sample_rate}


def load_protocol(protocol_csv: str, audio_dir: str,
                  sample_rate: int = TARGET_SR) -> list[Sample]:
    """Load samples described by a protocol CSV (columns: sample_id,
    audio_path, label, attack_id, speaker_id, domain)."""
    samples = []
    with open(protocol_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["audio_path"]
            if not os.path.isabs(path):
                path = os.path.join(audio_dir, path)
            audio, sr = load_audio(path, sample_rate)
            samples.append(Sample(
                sample_id=row["sample_id"],
                audio=audio,
                sample_rate=sr,
                label=int(row.get("label", -1)),
                attack_id=row.get("attack_id", "unknown"),
                speaker_id=row.get("speaker_id", "unknown"),
                domain=row.get("domain", "unknown"),
                audio_path=path,
            ))
    return samples


def split_by_label(samples: list[Sample]):
    bonafide = [s for s in samples if s.label == LABEL_BONAFIDE]
    spoof = [s for s in samples if s.label == LABEL_SPOOF]
    return bonafide, spoof
