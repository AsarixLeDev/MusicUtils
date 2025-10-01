# AudioRestorer – Music & Speech Denoising (Stems + Procedural Noise)

AudioRestorer is a PyTorch pipeline for **music/speech denoising** built around:
- **Stem‑separated music** (e.g., MUSDB18HQ, Moises) as clean targets
- **Procedural noise** (e.g., MUSAN, DEMAND) filtered to realistic artefacts
- A flexible **STFT mask** task (magnitude‑domain), unified utilities, and rich diagnostics

---

## Contents
- [Quickstart](#quickstart)
- [Composite Score (0..100)](#composite-score-0100)
- [Data Setup](#data-setup)
- [Training](#training)
- [Evaluation & Diagnostics](#evaluation--diagnostics)
- [Sanity Suite](#sanity-suite)
- [Export & Profiling (Optional)](#export--profiling-optional)
- [Folder Layout](#folder-layout)
- [FAQ](#faq)

---

## Quickstart

> Assumes Python 3.9+ and a recent PyTorch/torchaudio build with CUDA (if available).

```bash
# 1) Install project deps
pip install -r requirements.txt

# 2) (Optional) Run the sanity suite — no big data required
python tools/sanity_suite.py \
  --manifest soundrestorer/data/training/train.jsonl \
  --sr 48000 --mono

# 3) Train (full, curriculum, mixed music+noise)
python scripts/train.py --config configs/denoiser_flexible_v2.yaml

# 4) Inspect learning signals from audio_debug triads
python tools/learning_from_audio_debug.py \
  --root runs/<your_run>/logs/audio_debug --write-csv

# 5) Triad diagnostics (LSD, ΔSI, residual %, composite)
python tools/triad_diagnostics.py \
  --root runs/<your_run>/logs/audio_debug \
  --write-csv --summary-csv
````

On Windows PowerShell, you can keep the same commands; just use backslashes if you prefer:

```powershell
python tools\sanity_suite.py --manifest soundrestorer\data\training\train.jsonl --sr 48000 --mono
```

---

## Composite Score (0..100)

The **Composite Score** compresses several signals into a single human‑readable number:

* **ΔSI (dB)**: improvement in SI‑SDR vs the noisy input
  `ΔSI = SI(yhat, clean) − SI(noisy, clean)`
* **LSD improvement (dB)**: spectral quality gain
  `ΔLSD = LSD(noisy, clean) − LSD(yhat, clean)`  (higher is better)
* **Residual energy ratio**: proportion of noise not removed
  `r = ||yhat − clean||² / (||noisy − clean||² + 1e−12)` clipped to `[0, 1.5]`

We map each to `[0,1]` with gentle sigmoids and combine:

```
sigmoid(x; k, x0) = 1 / (1 + exp(−k · (x − x0)))

ΔSI_term  = sigmoid(ΔSI; 0.5,  1.0)     # ~0 at no change, →1 as ΔSI grows
LSD_term  = sigmoid(ΔLSD; 0.8,  1.0)    # sensitive in 0..3 dB improvement
Resid_term = 1 − min(1.0, r)            # 1 when residual ~0, →0 as residual dominates

Composite = 100 · [ 0.45·ΔSI_term + 0.35·LSD_term + 0.20·Resid_term ]
```

**Interpretation (rule‑of‑thumb):**

* 90–100: transparent or near‑studio‑clean
* 75–90: clearly improved, minor artefacts
* 60–75: workable improvement, audible artefacts
* <60: needs work or mismatch (data/task)

`tools/triad_diagnostics.py` and `tools/learning_from_audio_debug.py` compute per‑item and per‑epoch medians and write `CSV` reports.

---

## Data Setup

You can train from:

1. **Stems + mixture** (music): e.g., MUSDB18HQ / Moises
   Use stems as **clean**; `mixture = sum(stems)` helps validation and checks.
2. **Noise** libraries: MUSAN/DEMAND
   Filter to **plausible music/speech artefacts** (crowds, HVAC, hiss, rumble), not alarms.

### Manifest format

We use JSON Lines manifests:

```json
{"id":"track001","split":"train","clean":"/path/to/clean.wav","noisy":"/path/to/noisy.wav","sr":48000,"duration":12.34}
{"id":"track002","split":"val","clean":"/path/to/clean.wav","noisy":"/path/to/noisy.wav","sr":48000,"duration":9.87}
```

* For **stems** pipelines, prepare `clean` as your stem mix (e.g., `vocals + bass + drums + other`, or a target stem), and produce `noisy` by adding curated noise at random SNRs (the training loader can do this on the fly).
* Verify stem consistency:

```bash
python tools/audit_stems.py --root /path/to/stems_root --tol-db -40
```

---

## Training

Two configs you’ll use most:

* **Full training** (curriculum, mixed noise, callbacks):

  ```bash
  python scripts/train.py --config configs/denoiser_flexible_v2.yaml
  ```
* **Micro overfit** (one batch sanity; expect fast ΔSI > 0):

  ```bash
  python scripts/train.py --config configs/denoiser_overfit.yaml
  ```

During training, the loop writes:

* `runs/<run>/logs/audio_debug/epXXX/` → triads (`*_noisy.wav`, `*_yhat.wav`, `*_clean.wav`)
* `runs/<run>/logs/data_audit/epXXX/audit.csv` → dataset sample SNRs, silence %, etc.
* `runs/<run>/checkpoints/epoch_XXX.pt` → model/optimizer snapshots

**What to watch:**

* **SI‑SDR (dB)** on val: median ΔSI ≥ 0 dB (improving) over time
* **Residual %** ↓, **Composite** ↑ in audio_debug summaries
* No consistent **>25 dB clean/noisy SNR** on training data (too clean → slow learning)
* Data audit: few (ideally none) **>95% silence** items

---

## Evaluation & Diagnostics

**Learning trend from audio_debug:**

```bash
python tools/learning_from_audio_debug.py \
  --root runs/<run>/logs/audio_debug --write-csv
```

**Detailed per‑triad diagnostics (LSD, ΔSI, residual %, Composite):**

```bash
python tools/triad_diagnostics.py \
  --root runs/<run>/logs/audio_debug \
  --write-csv --summary-csv
```

**Batch inference + metrics CSV for a folder or manifest:**

```bash
python tools/infer_batch.py \
  --in /path/to/wavs_or_manifest \
  --checkpoint runs/<run>/checkpoints/epoch_XXX.pt \
  --config configs/denoiser_flexible_v2.yaml \
  --out out_wavs --metrics out_metrics.csv
```

---

## Sanity Suite

Run a one‑button sanity battery (no large datasets required):

```bash
python tools/sanity_suite.py \
  --manifest soundrestorer/data/training/train.jsonl \
  --sr 48000 --mono
```

It checks:

1. Manifest schema and paths
2. Noise filter coverage
3. SNR distribution sampling
4. Mixer round‑trip (SNR targets)
5. STFT/iSTFT unit test
6. Loss positivity & micro overfit (ΔSI > 0)
7. Callback smoke test
8. Tiny epoch run + CSV dump

See `docs/sanity_check_playbook.md` for thresholds and how to interpret.

---

## Export & Profiling (Optional)

**ONNX export:**

```bash
python scripts/export_onnx.py \
  --config configs/denoiser_flexible_v2.yaml \
  --checkpoint runs/<run>/checkpoints/epoch_XXX.pt \
  --out exports/denoiser.onnx --sr 48000 --length 96000
```

**Inference profiling (latency/memory):**

```bash
python scripts/profile_infer.py \
  --config configs/denoiser_flexible_v2.yaml \
  --checkpoint runs/<run>/checkpoints/epoch_XXX.pt \
  --device cuda:0 --secs 3.0 --runs 20
```

---

## Folder Layout

```
configs/
  denoiser_flexible_v2.yaml
  denoiser_overfit.yaml
docs/
  music_denoising_recipe.md
  sanity_check_playbook.md
scripts/
  train.py
  export_onnx.py
  profile_infer.py
soundrestorer/
  core/ ...         # trainer, factories
  tasks/ ...        # denoise_stft
  losses/ ...       # mrstft, l1_wave, sisdr_pos, etc.
  callbacks/ ...    # audio_debug, data_audit, perceptual_eval (optional)
  utils/ ...        # audio, signal, metrics, io (unified)
tools/
  sanity_suite.py
  triad_diagnostics.py
  learning_from_audio_debug.py
  audit_stems.py
  infer_batch.py
runs/
  <date_time>_<run_name>/
    checkpoints/
    logs/
```

---

## FAQ

**Q: Why do I see negative loss values?**
A: If you log raw **SI‑SDR (dB)** as a metric, it is often positive for “good” audio but can be negative for bad. Use a **positive loss** like `sisdr_pos` (e.g., `max(0, −SI)` or a ratio‑based variant), or simply plot **−SI‑SDR** to get an increasing, non‑negative curve.

**Q: The model seems slow to improve.**
A: Check the **data audit**: if many training items have **SNR > 25 dB**, reduce the “too clean” fraction via the **mixer** or curriculum; also verify you’re not feeding long **silent** segments.

**Q: Overfit doesn’t reach 0 loss. Is that expected?**
A: “Perfect 0” is rare with MR‑STFT + waveform losses due to windowing/OLA and numerical floors. For micro‑overfit, look for **ΔSI > +0.5..+1.0 dB** in a few hundred steps and **L1/MSE → very small**; that’s sufficient to validate learning.