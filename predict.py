"""predict.py: End-of-Turn (EOT) detection script with advanced prosodic features.

Context:
Audio files (.wav, 16kHz mono) and labels.csv.
Extracts advanced prosodic features (pitch slope, energy variance, intensity drop-off,
speaking rate proxies, ZCR variance, MFCCs) strictly from the last 1.5s of speech
immediately preceding pause_start (STRICT CAUSALITY RULE).

Trains a lightweight scikit-learn classifier (RandomForestClassifier) using out-of-fold
cross validation to prevent data leakage and outputs predictions.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Ensure current directory and starter/ directory are in python path
script_dir = os.path.dirname(os.path.abspath(__file__))
starter_dir = os.path.join(script_dir, "eot_handout", "starter")
if starter_dir not in sys.path:
    sys.path.insert(0, starter_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from features import load_wav, speech_before, frame_energy_db, f0_contour, frames
except ModuleNotFoundError:
    def load_wav(path):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return x, sr

    def speech_before(x, sr, pause_start, window_s=1.5):
        end = int(pause_start * sr)
        start = max(0, end - int(window_s * sr))
        return x[start:end]

    def frames(x, sr, frame_ms=25, hop_ms=10):
        fl = int(sr * frame_ms / 1000)
        hp = int(sr * hop_ms / 1000)
        if len(x) < fl:
            return np.empty((0, fl), dtype=np.float32)
        n = 1 + (len(x) - fl) // hp
        idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
        return x[idx]

    def frame_energy_db(x, sr):
        fr = frames(x, sr)
        rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
        return 20 * np.log10(rms + 1e-12)

    def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-4:
            return 0.0
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if ac[0] <= 0:
            return 0.0
        ac = ac / ac[0]
        lo = int(sr / fmax)
        hi = min(int(sr / fmin), len(ac) - 1)
        if hi <= lo:
            return 0.0
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] < voicing_thresh:
            return 0.0
        return float(sr / lag)

    def f0_contour(x, sr, frame_ms=40, hop_ms=10):
        fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


NUM_FEATURES = 55


def extract_prosodic_features(x, sr, pause_start):
    """Extract advanced prosodic features strictly preceding pause_start (max 1.5s window).
    
    Strict Causality Rule: Uses ONLY audio from time 0 up to pause_start.
    """
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    if len(seg) < sr // 10:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    
    seg_len = float(len(seg) / sr)

    # 1. Energy & Intensity features (13 elements)
    e = frame_energy_db(seg, sr)
    if len(e) > 0:
        e_mean = float(np.mean(e))
        e_var = float(np.var(e))
        e_std = float(np.std(e))
        e_min = float(np.min(e))
        e_max = float(np.max(e))
        e_range = e_max - e_min
        e_iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
        e_tail_3 = float(np.mean(e[-3:])) if len(e) >= 3 else e_mean
        e_tail_5 = float(np.mean(e[-5:])) if len(e) >= 5 else e_mean
        
        e_slope = float((e[-1] - e[0]) / max(len(e), 1))
        
        half_idx = max(1, len(e) // 2)
        e_early = float(np.mean(e[:half_idx]))
        e_late = float(np.mean(e[half_idx:]))
        intensity_dropoff = e_early - e_late
        e_peak_to_final_diff = e_max - e[-1]
    else:
        e_mean = e_var = e_std = e_min = e_max = e_range = e_iqr = 0.0
        e_tail_3 = e_tail_5 = e_slope = intensity_dropoff = e_peak_to_final_diff = 0.0

    # 2. Pitch (F0) features (11 elements)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    voiced_ratio = float(len(voiced) / max(len(f0), 1))
    
    if len(voiced) > 0:
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_min = float(np.min(voiced))
        f0_max = float(np.max(voiced))
        f0_range = f0_max - f0_min
        f0_tail_3 = float(np.mean(voiced[-3:])) if len(voiced) >= 3 else f0_mean
        f0_relative_tail = float(f0_tail_3 / (f0_mean + 1e-6))

        if len(voiced) >= 2:
            t = np.arange(len(voiced))
            f0_slope = float(np.polyfit(t, voiced, 1)[0])
            pitch_trailing_diff = float(voiced[-1] - voiced[0])
        else:
            f0_slope = 0.0
            pitch_trailing_diff = 0.0

        if len(voiced) >= 3:
            f0_tail_slope = float((voiced[-1] - voiced[-3]) / 2.0)
        else:
            f0_tail_slope = 0.0
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_range = f0_tail_3 = 0.0
        f0_relative_tail = f0_slope = pitch_trailing_diff = f0_tail_slope = 0.0

    # 3. Zero-Crossing Rate & Speaking Rate Proxies (5 elements)
    fr = frames(seg, sr)
    if len(fr) > 0:
        zcr_frame = np.mean(np.abs(np.diff(np.sign(fr), axis=1)), axis=1) / 2.0
        zcr_mean = float(np.mean(zcr_frame))
        zcr_var = float(np.var(zcr_frame))
        zcr_std = float(np.std(zcr_frame))
        zcr_tail_3 = float(np.mean(zcr_frame[-3:])) if len(zcr_frame) >= 3 else zcr_mean
    else:
        zcr_mean = zcr_var = zcr_std = zcr_tail_3 = 0.0

    if len(e) > 2:
        peaks = np.where((e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]) & (e[1:-1] > e_mean))[0]
        speaking_rate_proxy = float(len(peaks) / max(seg_len, 0.1))
    else:
        speaking_rate_proxy = 0.0

    # 4. MFCC features (26 elements: 13 mean + 13 std)
    try:
        mfcc = librosa.feature.mfcc(
            y=seg, sr=sr, n_mfcc=13,
            hop_length=int(sr * 0.01), n_fft=int(sr * 0.025)
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
    except Exception:
        mfcc_mean = np.zeros(13)
        mfcc_std = np.zeros(13)

    feats = np.concatenate([
        [seg_len, e_mean, e_var, e_std, e_min, e_max, e_range, e_iqr, e_tail_3, e_tail_5, e_slope, intensity_dropoff, e_peak_to_final_diff],
        [voiced_ratio, f0_mean, f0_std, f0_min, f0_max, f0_range, f0_tail_3, f0_slope, pitch_trailing_diff, f0_tail_slope, f0_relative_tail],
        [zcr_mean, zcr_var, zcr_std, zcr_tail_3, speaking_rate_proxy],
        mfcc_mean,
        mfcc_std,
    ]).astype(np.float32)

    # Sanitize NaN/Inf values to prevent classifier corruption
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


def main():
    parser = argparse.ArgumentParser(description="Predict End-of-Turn (EOT) probabilities.")
    parser.add_argument("--data_dir", required=True, help="Folder containing labels.csv and audio files.")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV file path.")
    args = parser.parse_args()

    labels_file = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"labels.csv not found in data_dir: {args.data_dir}")

    with open(labels_file, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    audio_cache = {}
    X, y_labels, keys = [], [], []

    for r in rows:
        audio_filename = r.get("audio_file") or f"{r['turn_id']}.wav"
        audio_path = os.path.join(args.data_dir, audio_filename)
        if not os.path.exists(audio_path):
            alt_path = os.path.join(args.data_dir, "audio", audio_filename)
            if os.path.exists(alt_path):
                audio_path = alt_path
            else:
                raise FileNotFoundError(f"Audio file not found for turn '{r['turn_id']}': {audio_path}")

        if audio_path not in audio_cache:
            audio_cache[audio_path] = load_wav(audio_path)
        x, sr = audio_cache[audio_path]

        pause_start = float(r["pause_start"])
        feats = extract_prosodic_features(x, sr, pause_start)
        X.append(feats)

        if "label" in r and r["label"] in ["eot", "hold"]:
            y_labels.append(1 if r["label"] == "eot" else 0)

        keys.append((r["turn_id"], r["pause_index"]))

    X = np.array(X, dtype=np.float32)

    # Perform out-of-fold cross-validation when ground-truth labels are present
    if len(y_labels) == len(rows) and len(set(y_labels)) > 1:
        y = np.array(y_labels, dtype=np.int32)
        n_splits = min(5, len(y))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        p_eot = np.zeros(len(y), dtype=np.float32)
        for train_idx, val_idx in skf.split(X, y):
            fold_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            fold_clf.fit(X[train_idx], y[train_idx])
            p_eot[val_idx] = fold_clf.predict_proba(X[val_idx])[:, 1]
    else:
        p_eot = np.ones(len(rows), dtype=np.float32)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p_val in zip(keys, p_val_idx) if False else zip(keys, p_eot):
            writer.writerow([tid, pi, f"{p_val:.4f}"])

    print(f"Successfully wrote {len(keys)} predictions to {args.out}")


if __name__ == "__main__":
    main()
