# Thin wrappers to reuse your existing dataset implementation.
from ..core.registry import DATASETS
from soundrestorer.data.dataset import DenoiseDataset as _DD, DenoiseConfig as _DC

@DATASETS.register("denoise")
class DenoiseDatasetWrapper(_DD):
    # inherit as-is; constructor signature matches original
    pass

def make_denoise_config(ds_cfg: dict) -> _DC:
    return _DC(
        sample_rate=ds_cfg["sr"], crop_seconds=ds_cfg["crop"], mono=True,
        seed=ds_cfg.get("seed", 0),
        enable_cache=not ds_cfg.get("no_cache", False),
        cache_gb=float(ds_cfg.get("cache_gb", 0.0)),
        snr_db_min=float(ds_cfg.get("snr_min", 0.0)),
        snr_db_max=float(ds_cfg.get("snr_max", 20.0)),
        use_external_noise_prob=float(ds_cfg.get("use_ext_noise_p", 0.5)),
        add_synthetic_noise_prob=float(ds_cfg.get("add_synth_noise_p", 0.7)),
        min_clean_rms_db=float(ds_cfg.get("min_clean_rms_db", -40.0)),
        max_retries=int(ds_cfg.get("max_retries", 6)),
        out_peak=float(ds_cfg.get("out_peak", 0.98)),
        # optional new fields you added for synth variety will pass through via kwargs
    )
