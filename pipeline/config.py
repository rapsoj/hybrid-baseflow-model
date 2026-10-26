"""Load config.yaml, apply command line overrides and derive step counts."""
import copy

import yaml


def _set_nested(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section '{k}' in override '{dotted_key}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key in override '{dotted_key}'")
    node[keys[-1]] = value


def load_config(path="config.yaml", borehole: str = "chilgrove", overrides=()) -> dict:
    """
    Returns the config with:
      cfg["borehole_key"], cfg["borehole"]  - the selected borehole's settings
      cfg["paths"]                          - paths with {borehole} substituted
      cfg["derived"]                        - step counts computed from the data section
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_nested(cfg, key.strip(), yaml.safe_load(raw))

    if borehole not in cfg["boreholes"]:
        raise ValueError(f"Unknown borehole '{borehole}'. Configured: {sorted(cfg['boreholes'])}")
    cfg["borehole_key"] = borehole
    cfg["borehole"] = cfg["boreholes"][borehole]
    prefix = cfg["borehole"]["geosmart_prefix"].lower()
    cfg["paths"] = {k: v.format(borehole=borehole, geosmart_prefix=prefix) for k, v in cfg["paths"].items()}
    cfg["derived"] = _derive(cfg)
    return cfg


def _derive(cfg: dict) -> dict:
    d = cfg["data"]
    step = int(d["step_minutes"])
    if 1440 % step:
        raise ValueError("data.step_minutes must divide a day")
    per_day = 1440 // step

    bin_minutes = int(d["gefs"]["bin_hours"]) * 60
    if bin_minutes % step:
        raise ValueError("data.gefs.bin_hours must be a multiple of data.step_minutes")
    out_step = int(d["output_step_minutes"])
    if out_step % step or bin_minutes % out_step:
        raise ValueError("data.output_step_minutes must be a multiple of step_minutes and divide gefs.bin_hours")
    token_steps = bin_minutes // out_step  # output values emitted per decoder token

    hh, mm = (int(x) for x in str(d["origin_time"]).split(":"))
    if (hh * 60 + mm) % bin_minutes:
        raise ValueError(
            "data.origin_time must fall on a GEFS bin boundary (multiple of gefs.bin_hours from 00:00) "
            "so decoder tokens line up exactly with GEFS accumulation windows"
        )

    if d.get("target", "total_flow") not in ("total_flow", "runoff", "baseflow"):
        raise ValueError("data.target must be total_flow, runoff or baseflow")
    if d["target_mode"] not in ("absolute", "delta_log_last", "anomaly", "delta_log_last_anomaly"):
        raise ValueError("data.target_mode must be absolute, delta_log_last, anomaly or delta_log_last_anomaly")

    history_steps = int(d["history_days"]) * per_day
    out_per_day = 1440 // out_step
    horizon_fine_steps = int(d["horizon_days"]) * per_day
    horizon_steps = int(d["horizon_days"]) * out_per_day
    decoder_steps = int(d["decoder_days"]) * out_per_day
    if int(d["horizon_days"]) % 7:
        raise ValueError("data.horizon_days must be a whole number of weeks (metrics are reported per forecast week)")
    if decoder_steps < horizon_steps:
        raise ValueError("data.decoder_days must be >= data.horizon_days")
    if horizon_steps % token_steps or decoder_steps % token_steps:
        raise ValueError("horizon_days and decoder_days must be whole multiples of gefs.bin_hours")
    if history_steps % int(cfg["model"]["encoder_patch_steps"]):
        raise ValueError("history steps must be divisible by model.encoder_patch_steps")
    if cfg["model"]["d_model"] % cfg["model"]["n_heads"]:
        raise ValueError("model.d_model must be divisible by model.n_heads")

    return {
        "steps_per_day": out_per_day,  # output steps per day
        "history_steps": history_steps,
        "horizon_fine_steps": horizon_fine_steps,  # horizon in step_minutes steps (target window length on the HDE grid)
        "output_ratio": out_step // step,
        "horizon_steps": horizon_steps,
        "decoder_steps": decoder_steps,
        "token_steps": token_steps,
        "n_decoder_tokens": decoder_steps // token_steps,
        "n_output_tokens": horizon_steps // token_steps,
        "origin_minutes": hh * 60 + mm,
    }


def snapshot(cfg: dict) -> dict:
    """Config subset stored next to weights and results (what defines a trained model)."""
    keys = ("borehole_key", "borehole", "data", "model", "training", "ablation")
    out = {k: copy.deepcopy(cfg[k]) for k in keys}
    out["derived"] = dict(cfg["derived"])
    return out
