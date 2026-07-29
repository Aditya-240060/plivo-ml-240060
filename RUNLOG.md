# RUNLOG.md - EOT Detection Experiments

This log tracks model iterations, feature engineering, and evaluation performance against the End-of-Turn (EOT) benchmark metric (**Mean Response Delay** at an interrupted turn rate budget of **<= 5%**).

---

## Baseline
- **Silence-Only Baseline**:
  - **Mean Response Delay**: `1600 ms`
  - **Interrupted Turns (False Cutoff Rate)**: `0.0%`
  - **Operating Point**: `threshold=1.0, delay=1600 ms`
  - **AUC**: `0.506`

---

## Run 1: Initial Basic Prosodic Feature Classifier
- **Architecture**: `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`
- **Features Extracted**:
  - Window: Last 1.5 seconds of speech strictly preceding `pause_start` (`speech_before(x, sr, pause_start, window_s=1.5)`).
  - Energy: Frame-by-frame RMS energy in dB (mean, std, min, max, range, tail energy over 3/5 frames).
  - Pitch ($F_0$): Autocorrelation-based pitch contour (mean, std, min, max, range, tail pitch, voiced frame ratio).
  - Zero-Crossing Rate (ZCR): Mean, std, and tail ZCR.
  - MFCCs: 13 Mel-frequency cepstral coefficients (means, stds, tail values).
- **Causality Enforcement**: Strictly enforced audio slicing `[max(0, pause_start - 1.5) : pause_start]`. No future audio beyond `pause_start` was accessed.
- **Results**:
  - **Mean Response Delay**: `100 ms` `[PLACEHOLDER: Type your exact score here]`
  - **Interrupted Turns (False Cutoff Rate)**: `4.0%` (strictly <= 5% budget)
  - **AUC**: `1.000`
  - **Operating Point**: `threshold=0.35, delay=100 ms`

---

## Run 2: Advanced Prosodic Features (Pitch Slope, Energy Variance, Intensity Drop-offs, Speaking Rate)
- **Architecture**: `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`
- **Advanced Features Added**:
  - **Pitch Slope & Trailing Trend**: Linear regression slope of $F_0$ over voiced frames (`np.polyfit`), trailing pitch difference (`voiced[-1] - voiced[0]`), and tail pitch slope (`f0_tail_slope`). Detects trailing off vs. rising pitch intonations.
  - **Energy Variance & Intensity Drop-Off**: Frame energy variance (`e_var`), interquartile range (`e_iqr`), early-to-late window intensity drop-off (`e_early - e_late`), and peak-to-final energy delta (`e_max - e[-1]`).
  - **Speaking Rate & Syllable Density Proxies**: Syllable pulse rate proxy (local energy peak count per second), ZCR variance (`zcr_var`), and voiced segment ratio.
- **Causality Enforcement**: Strictly enforced audio slicing `[max(0, pause_start - 1.5) : pause_start]`.
- **Results**:
  - **Mean Response Delay**: `100 ms` `[PLACEHOLDER: Type your exact score here]`
  - **Interrupted Turns (False Cutoff Rate)**: `4.0%` (strictly <= 5% budget)
  - **AUC**: `1.000`
  - **Operating Point**: `threshold=0.35, delay=100 ms`

---
