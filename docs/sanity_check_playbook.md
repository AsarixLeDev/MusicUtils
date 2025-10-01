# Sanity‑Check Playbook

This guide explains what to run first, what numbers to expect, and how to debug when things look off.

---

## 1) What to run first

```bash
# Manifests & basic pipeline health
python tools/sanity_suite.py \
  --manifest soundrestorer/data/training/train.jsonl \
  --sr 48000 --mono
````

It performs:

1. Manifest schema & path validation
2. Noise filter coverage
3. SNR distribution sampling
4. Mixer round‑trip SNR checks
5. STFT/iSTFT roundtrip (MSE < 1e−6)
6. Loss positivity test & micro overfit (ΔSI > 0 in a few hundred steps)
7. Callback smoke test
8. Tiny training epoch + CSV dump

**Exit code 0** = all good. It prints any failed steps with hints.

---

## 2) Thresholds to watch

* **STFT/iSTFT**: MSE < 1e−6
* **Mixer SNR**: measured SNR within ±0.6 dB of target
* **Audio Debug**:

  * **Residual energy %**: trending down per epoch
  * **Composite**: medians trending up, ideally → 70–80+
* **Data Audit**:

  * “Too clean” items (SNR > 25 dB) should be a minority
  * Silence (>95%) should be rare

---

## 3) Interpreting negative values

* **Raw SI‑SDR (dB)** can be negative when outputs are poor; that’s normal.
  For a **positive loss curve**, log **−SI‑SDR** or use a **positive variant** (e.g., clamp/ratio).

---

## 4) Common issues & fixes

* **“No improvement” in ΔSI**
  → Reduce fraction of clean items; widen SNR range; verify mask floors/limits; confirm noise actually injected.
* **“I hear no noise in _noisy”**
  → Inspect `data_audit` WAVs and SNR metrics; raise noise gain window; ensure non‑silent crops.
* **Overfit plateau**
  → Re‑enable waveform losses; slightly increase LR; reduce crop length; ensure mask bias init isn’t suppressing changes.

---

## 5) Minimal targets before long runs

* Micro overfit: ΔSI ≥ +0.5..+1.0 dB within ~300–600 steps
* Sanity suite: all steps pass
* Audio debug (first few epochs): visibly positive ΔSI items; residual % decreasing