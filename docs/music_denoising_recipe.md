# Music Denoising Recipe (Stems + Procedural Noise)

This note walks through a recommended setup for **music denoising** using **stems** as clean targets and **procedural noise** as mix‑ins.

---

## 1) Data Preparation

### 1.1 Stems & Mixture
- Use MUSDB18HQ/Moises (or similar). For each track:
  - Pick your **clean target**: mixture of stems (e.g., all stems), or a specific stem.
  - Optionally verify: `mixture ≈ sum(stems)` within tolerance:
    ```bash
    python tools/audit_stems.py --root /path/to/stems --tol-db -40
    ```
  - Create a **manifest** mapping `clean` and `noisy` fields (noisy can be same as clean if you plan to add noise **on the fly**; your loader/mixer can inject noise during training).

### 1.2 Noise Libraries
- Pull subsets of **MUSAN** and **DEMAND** that sound plausible in music/speech contexts:
  - crowd chatter, room tone, HVAC, tape hiss, rumble, mild clicks
  - avoid unrealistic alarms, sirens, sudden transients (unless you want robustness to them)
- Store noise clips in a separate folder; the mixer will sample them.

### 1.3 Manifest Rows
```json
{"id":"song001","split":"train","clean":"/data/clean/song001.wav","noisy":"/data/clean/song001.wav","sr":48000,"duration":12.8}
````

* For on‑the‑fly mixing, `noisy` can point to the same file as `clean` (the loader adds noise).

---

## 2) Configs

* **Full training**: `configs/denoiser_flexible_v2.yaml`
* **Micro overfit**:  `configs/denoiser_overfit.yaml`

These define:

* STFT mask task (magnitude domain; `mask_floor`/`mask_limit`)
* Losses (e.g., MR‑STFT, L1 waveform, SI‑SDR positive variant)
* Curriculum and data audit settings

---

## 3) Training

```bash
python scripts/train.py --config configs/denoiser_flexible_v2.yaml
```

**Watch for:**

* Validation **ΔSI (median)** ≥ 0 dB over epochs
* **Residual energy %** trending down in `tools/learning_from_audio_debug.py`
* **Composite Score** medians trending up
* Triads (`audio_debug`) audibly cleaner by epoch ~5–10

**Common pitfalls:**

* Dataset too clean → little to learn (many SNR > 25 dB)
* Silent segments dominating crops
* Mask constraints too tight (`mask_floor` too high, or `mask_limit` too low)
* LR schedule too aggressive for your batch size

---

## 4) Evaluation

### 4.1 Triad Diagnostics (per‑epoch)

```bash
python tools/triad_diagnostics.py \
  --root runs/<run>/logs/audio_debug \
  --write-csv --summary-csv
```

Outputs:

* `triad_diagnostics.csv` with per‑item SI‑SDR, LSD, residual %, Composite
* `learning_summary.csv` medians per epoch

### 4.2 Batch Inference

```bash
python tools/infer_batch.py \
  --in /path/to/folder_or_manifest \
  --checkpoint runs/<run>/checkpoints/epoch_XXX.pt \
  --config configs/denoiser_flexible_v2.yaml \
  --out out_wavs --metrics out_metrics.csv
```

### 4.3 Listening Tests

* Curate a small “hard set” (hiss + reverb + crowd).
* Compare `noisy` vs `yhat` vs `clean` quickly in a DAW or with a playlist.

---

## 5) Tips

* **Inference overlap**: Slightly higher STFT overlap at inference can reduce musical noise.
* **Post‑filters**: Simple spectral floors and limits help stability on out‑of‑domain inputs.
* **Architectures**: Start with UNet; consider adding small recurrent or attention blocks in the bottleneck for longer‑range context if CPU/GPU budget allows.
* **Curriculum**: Start 0–20 dB SNR, then open up (−2..20 dB) once stable.