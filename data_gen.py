"""Synthetic demo data generator for the DeepGuard-Audio backend.

Real evaluation should point at ASVspoof 2021/5 audio + a protocol file.  This
module produces a small, self-contained bonafide/spoof dataset so the whole
pipeline runs end-to-end and produces non-trivial metrics out of the box.

The synthesised "speech" is a sequence of voiced F0-harmonic segments with
modulated formants and noise.  "Spoof" variants inject the kinds of artefacts
that anti-spoofing detectors exploit: sub-band quantisation noise, phase
discontinuities, vocoder buzz (spectral flatness in high bands), and
unnatural pitch continuity.

Output layout::

    data/synth/
        wav/<sample_id>.wav
        protocol.csv   # columns: sample_id,label,attack_id,speaker_id,domain
"""
from __future__ import annotations

import os
import csv
import numpy as np

try:  # package context
    from .audio_io import save_audio, TARGET_SR
except ImportError:  # script context
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from audio_io import save_audio, TARGET_SR  # type: ignore

LABEL_BONAFIDE = 0
LABEL_SPOOF = 1


def _voice_segment(rng, n, f0, formants, sr):
    """A short voiced segment: harmonic stack through formant resonators."""
    t = np.arange(n) / sr
    # pitch contour with slight jitter + vibrato
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
    f0_inst = f0 * vib * (1.0 + 0.01 * rng.standard_normal(n))
    phase = 2 * np.pi * np.cumsum(f0_inst) / sr
    src = np.zeros(n)
    for h in range(1, 8):
        src += (1.0 / h) * np.sin(h * phase)
    # formant filtering via simple parallel resonators (one-pole, one-zero approx)
    out = src.copy()
    for _fc, bw, gain in formants:
        # simple bandpass-ish emphasis around fc
        a = np.exp(-2 * np.pi * bw / sr)
        b = 1 - a
        # bandpass by differencing two lowpass on the analytic-ish signal
        lp = lfilter_one_pole(src, b, a)
        out = out + gain * lp
    # breathiness noise modulated by amplitude
    amp = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    out += 0.02 * amp * rng.standard_normal(n)
    return out


def lfilter_one_pole(x, b, a):
    y = np.empty_like(x, dtype=np.float64)
    y[0] = b * x[0]
    for i in range(1, len(x)):
        y[i] = b * x[i] + a * y[i - 1]
    return y


def _generate_utterance(rng, sr=TARGET_SR, duration=3.0, spoof=False, attack_id="clean"):
    """Generate one utterance.  If ``spoof`` inject artefacts keyed by attack_id."""
    n = int(duration * sr)
    rng_local = rng
    # 4-6 voiced segments separated by short pauses
    n_seg = rng_local.integers(4, 7)
    bounds = np.sort(rng_local.choice(np.arange(1, n_seg) * (n // n_seg),
                                      size=n_seg - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [n]])
    sig = np.zeros(n, dtype=np.float64)
    formant_sets = [
        [(700, 90, 1.0), (1100, 120, 0.6), (2500, 180, 0.3)],
        [(500, 80, 1.0), (1600, 130, 0.5), (2700, 200, 0.25)],
        [(300, 70, 1.0), (1900, 140, 0.45), (2400, 160, 0.3)],
    ]
    for s in range(n_seg):
        a, b = bounds[s], bounds[s + 1]
        seg_len = b - a
        if seg_len < int(0.08 * sr):
            continue
        f0 = rng_local.uniform(90, 180)
        formants = formant_sets[rng_local.integers(0, len(formant_sets))]
        seg = _voice_segment(rng_local, seg_len, f0, formants, sr)
        # apply a gentle onset/offset envelope
        env = np.ones(seg_len)
        ramp = int(0.02 * sr)
        if ramp > 0 and ramp < seg_len:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        sig[a:b] = seg * env

    # global normalisation
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8

    if spoof:
        sig = _inject_spoof_artefacts(sig, sr, attack_id, rng_local)
    return sig.astype(np.float32)


def _inject_spoof_artefacts(sig, sr, attack_id, rng):
    """Inject DeepFake-style artefacts that detectors can latch onto."""
    method = attack_id if attack_id != "clean" else "voc"
    if method in ("voc", "tts", "vc"):
        # vocoder buzz: sharpen high-frequency periodicity + spectral tilt
        from scipy.signal import fftconvolve
        # add a weak periodic buzz at a sub-multiple
        t = np.arange(len(sig)) / sr
        buzz = np.sign(np.sin(2 * np.pi * 110 * t)) * 0.05
        sig = sig + buzz
        # quantise phase-ish: high-band spectral flattening via comb emphasis
        sig = fftconvolve(sig, np.array([1.0, -0.6, 0.3, -0.15]), mode="same")
    elif method in ("codec", "neural"):
        # sub-band quantisation noise (neural vocoder residual)
        from scipy.signal import fftconvolve
        noise = rng.standard_normal(len(sig)) * 0.04
        sig = sig + noise
        sig = fftconvolve(sig, np.array([0.4, 0.3, 0.2, 0.1]), mode="same")
    elif method == "phase":
        # phase discontinuity: random segment sign flips at boundaries
        n_flips = rng.integers(3, 8)
        idx = np.sort(rng.choice(len(sig) - 1, size=n_flips, replace=False))
        idx = np.concatenate([[0], idx, [len(sig)]])
        sign = 1.0
        for k in range(len(idx) - 1):
            sig[idx[k]:idx[k + 1]] *= sign
            sign *= -1
        sig *= 0.7
    # generic spoof marker: very slight spectral flattening in high band
    from scipy.signal import lfilter
    sig = lfilter([1.0, -0.3], [1.0, -0.1], sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    return sig.astype(np.float32)


def generate_dataset(out_dir: str, n_bonafide: int = 20, n_spoof: int = 20,
                     sr: int = TARGET_SR, seed: int = 1234, duration: float = 3.0,
                     speakers: int = 4) -> str:
    """Generate a small bonafide+spoof dataset and a protocol CSV.

    Returns the path to the protocol CSV.
    """
    rng = np.random.default_rng(seed)
    wav_dir = os.path.join(out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    protocol = os.path.join(out_dir, "protocol.csv")

    spoof_kinds = ["voc", "tts", "vc", "codec", "neural", "phase"]
    rows = []
    for i in range(n_bonafide):
        sid = f"bona_{i:04d}"
        spk = f"spk{rng.integers(0, speakers)}"
        wav = _generate_utterance(rng, sr, duration, spoof=False)
        save_audio(os.path.join(wav_dir, f"{sid}.wav"), wav, sr)
        rows.append([sid, f"{sid}.wav", LABEL_BONAFIDE, "human", spk, "clean"])

    for i in range(n_spoof):
        sid = f"spoof_{i:04d}"
        spk = f"spk{rng.integers(0, speakers)}"
        attack_id = spoof_kinds[i % len(spoof_kinds)]
        wav = _generate_utterance(rng, sr, duration, spoof=True, attack_id=attack_id)
        save_audio(os.path.join(wav_dir, f"{sid}.wav"), wav, sr)
        rows.append([sid, f"{sid}.wav", LABEL_SPOOF, attack_id, spk, "synth"])

    with open(protocol, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "audio_path", "label", "attack_id",
                    "speaker_id", "domain"])
        w.writerows(rows)
    return protocol


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "synth"))
    ap.add_argument("--n-bonafide", type=int, default=20)
    ap.add_argument("--n-spoof", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    proto = generate_dataset(a.out, a.n_bonafide, a.n_spoof, seed=a.seed)
    print(f"Generated dataset -> protocol: {proto}")
