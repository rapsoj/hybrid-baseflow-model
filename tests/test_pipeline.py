import numpy as np
import pandas as pd
import pytest
import torch

from pipeline import data as D
from pipeline.config import load_config
from pipeline.evaluate import classification_metrics, pinball
from pipeline.model import FlowTransformer, pinball_loss

CFG_PATH = "config.yaml"


def small_cfg(**extra):
    over = ["model.d_model=16", "model.n_heads=2", "model.dim_feedforward=32", "training.epochs=1",
            "training.batch_size=8", "training.device=cpu"] + [f"{k}={v}" for k, v in extra.items()]
    return load_config(CFG_PATH, "chilgrove", over)


def geosmart_frame(issue, n_members=10, n_days=14, level=None):
    """One GeoSmart file: ID 0 history ending at issue, members with daily values at 00:00 from D+1."""
    hist_t = pd.date_range(issue - pd.Timedelta(hours=1), issue, freq="15min")
    rows = [(0, t, 50.0 if level is None else level) for t in hist_t]
    day1 = issue.normalize() + pd.Timedelta(days=1)
    for m in range(1, n_members + 1):
        for d in range(n_days):
            rows.append((m, day1 + pd.Timedelta(days=d), 50.0 + m + d if level is None else level))
    df = pd.DataFrame(rows, columns=["TimeSeriesID", "Date", "Level"])
    df["Date"] = df["Date"].dt.strftime("%d-%m-%Y %H:%M")
    return df


def test_geosmart_excludes_id0_and_computes_member_std():
    issue = pd.Timestamp("2020-03-10 09:15")
    fc = D.parse_geosmart_file(geosmart_frame(issue), 0.9, 2, 0)
    assert fc.anchor_time == issue and fc.anchor_level == 50.0
    members = np.arange(1, 11)
    np.testing.assert_allclose(fc.mean[0], 50 + members.mean() )
    np.testing.assert_allclose(fc.std[0], members.std(ddof=1))
    assert fc.knot_times[0] == np.datetime64("2020-03-11T00:00")
    assert len(fc.mean) == 14


def test_geosmart_drops_days_with_reduced_ensemble():
    issue = pd.Timestamp("2020-03-10 09:15")
    df = geosmart_frame(issue)
    d = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M")
    late = (df["TimeSeriesID"] > 5) & (d >= pd.Timestamp("2020-03-20"))
    fc = D.parse_geosmart_file(df[~late], 0.9, 2, 0)
    assert pd.Timestamp(fc.knot_times[-1]) < pd.Timestamp("2020-03-20")


def test_geosmart_interpolates_by_valid_time():
    issue = pd.Timestamp("2020-03-10 09:15")
    fc = D.parse_geosmart_file(geosmart_frame(issue), 0.9, 2, 0)
    m, s = fc.at(pd.DatetimeIndex([issue, pd.Timestamp("2020-03-11 00:00"), pd.Timestamp("2099-01-01")]))
    assert m[0] == 50.0 and s[0] == 0.0
    assert m[1] == pytest.approx(fc.mean[0])
    assert m[2] == pytest.approx(fc.mean[-1])


def test_purged_split_has_no_overlap():
    origins = pd.date_range("2020-01-01 12:00", periods=500, freq="D")
    gap = pd.Timedelta(days=28)
    sp = D.purged_chronological_split(origins, gap, 0.7, 0.15)
    for a, b in (("train", "val"), ("val", "test")):
        assert len(sp[a]) and len(sp[b])
        assert origins[sp[b][0]] - origins[sp[a][-1]] > gap
    assert len(sp["val"]) > 50 and len(sp["test"]) > 50


def _synthetic(cfg, n_days=120):
    step = cfg["data"]["step_minutes"]
    grid = pd.date_range("2020-01-01", periods=n_days * 24 * 60 // step, freq=f"{step}min")
    rng = np.random.default_rng(0)
    hde = pd.DataFrame({"rain": rng.gamma(0.2, 0.3, len(grid)),
                        "gw": 50 + np.cumsum(rng.normal(0, 0.01, len(grid))),
                        "flow": np.abs(rng.normal(1, 0.3, len(grid)))}, index=grid)
    bin_h = cfg["data"]["gefs"]["bin_hours"]
    n_bins = int(cfg["derived"]["n_decoder_tokens"]) + 4
    gefs, geo = {}, []
    for init in pd.date_range(grid[0].normalize(), grid[-1].normalize(), freq="D"):
        idx = pd.DatetimeIndex([init + pd.Timedelta(hours=bin_h * (i + 1)) for i in range(n_bins)])
        # value encodes the init day so the test can recover which forecast was used
        val = (init - grid[0]).days
        gefs[init] = pd.DataFrame({"mean_mm": float(val), "std_mm": 0.5}, index=idx)
        issue = init + pd.Timedelta(hours=9, minutes=15)
        geo.append(D.parse_geosmart_file(geosmart_frame(issue, level=float(val), n_days=cfg["data"]["horizon_days"] + 2), 0.9, 2, 0))
    return hde, gefs, geo, grid[0]


def test_no_temporal_leakage_in_decoder_inputs():
    cfg = small_cfg(**{"data.history_days": 2, "data.horizon_days": 7, "data.decoder_days": 7})
    hde, gefs, geo, t_start = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    assert s.has_gefs.all() and s.has_geosmart.all()
    bin_td = pd.Timedelta(hours=cfg["data"]["gefs"]["bin_hours"])
    for i, t0 in enumerate(s.origins):
        gefs_init = t_start.normalize() + pd.Timedelta(days=int(round(np.expm1(s.dec[i, 0, 0]))))
        assert gefs_init + bin_td <= t0                       # first bin of init is already complete at t0
        assert t0 - gefs_init <= pd.Timedelta(hours=cfg["data"]["gefs"]["max_age_hours"])
        geo_day = t_start.normalize() + pd.Timedelta(days=int(round(s.dec[i, 0, 2])) - 0)
        assert geo_day + pd.Timedelta(hours=9, minutes=15) <= t0   # GeoSmart issued before origin
        # encoder and target windows are disjoint in time
        assert s.window_start[i] < t0 < s.window_end[i]
        assert s.y.shape[1] == cfg["derived"]["horizon_steps"]


def test_target_follows_origin():
    cfg = small_cfg(**{"data.history_days": 2, "data.horizon_days": 7, "data.decoder_days": 7})
    hde, gefs, geo, _ = _synthetic(cfg)
    hde["flow"] = np.arange(len(hde), dtype=float)
    s = D.build_samples(cfg, hde, gefs, geo)
    i0 = hde.index.get_loc(s.origins[0])
    r = cfg["derived"]["output_ratio"]
    assert s.y[0, 0] == np.mean(np.arange(i0 + 1, i0 + 1 + r))
    assert s.y[0, -1] == np.mean(np.arange(i0 + 1 + (len(s.y[0]) - 1) * r, i0 + 1 + len(s.y[0]) * r))


def test_model_shape_and_noncrossing():
    cfg = small_cfg()
    dv = cfg["derived"]
    m = FlowTransformer(4, 6, dv["history_steps"], dv["n_decoder_tokens"], dv["n_output_tokens"], dv["token_steps"],
                        cfg["model"]["quantiles"], d_model=16, n_heads=2, dim_feedforward=32)
    out = m(torch.randn(2, dv["history_steps"], 4), torch.randn(2, dv["n_decoder_tokens"], 6))
    assert out.shape == (2, dv["horizon_steps"], len(cfg["model"]["quantiles"]))
    assert (out[..., 1:] >= out[..., :-1]).all()
    q = torch.tensor(cfg["model"]["quantiles"])
    assert pinball_loss(out, torch.zeros(2, dv["horizon_steps"]), q).ndim == 0


def test_model_catchment_embedding():
    cfg = small_cfg()
    dv = cfg["derived"]
    cfg["n_catchments"] = 3
    from pipeline.train import build_model

    m = build_model(cfg, 4, 6)
    out = m(
        torch.randn(2, dv["history_steps"], 4),
        torch.randn(2, dv["n_decoder_tokens"], 6),
        torch.tensor([0, 2]),
    )
    assert out.shape == (2, dv["horizon_steps"], len(cfg["model"]["quantiles"]))
    with pytest.raises(ValueError, match="catchment_id is required"):
        m(torch.randn(2, dv["history_steps"], 4), torch.randn(2, dv["n_decoder_tokens"], 6))


def test_model_geosmart_skip():
    cfg = small_cfg()
    dv = cfg["derived"]
    from pipeline.train import build_model

    torch.manual_seed(0)
    plain = build_model(cfg, 4, 6)
    skip = build_model(cfg, 4, 6, geosmart_skip_index=5)
    skip.load_state_dict(plain.state_dict(), strict=False)
    enc, dec = torch.randn(2, dv["history_steps"], 4), torch.randn(2, dv["n_decoder_tokens"], 6)
    plain.eval(), skip.eval()
    assert torch.allclose(plain(enc, dec), skip(enc, dec))  # zero-initialised skip changes nothing
    skip.geosmart_skip.weight.data.fill_(0.5)
    assert not torch.allclose(plain(enc, dec), skip(enc, dec))
    skip.train()
    skip.geosmart_skip.weight.data.zero_()
    skip(enc, dec).sum().backward()
    assert skip.geosmart_skip.weight.grad.abs().sum() > 0  # the zero-initialised skip still learns


def test_geosmart_representation():
    n, members, tokens = 2, 3, 4
    levels = np.arange(members)[None, :, None] + np.arange(tokens)[None, None, :] + np.zeros((n, 1, 1))
    origin = np.full((n, members), 10.0, dtype=np.float32)
    s = D.SampleSet(
        origins=pd.date_range("2024-01-01", periods=n), enc=np.zeros((n, 1, 4), np.float32),
        dec=np.zeros((n, tokens, len(D.DEC_CHANNELS)), np.float32), y=np.zeros((n, 1), np.float32),
        last_flow=np.zeros(n, np.float32), anchor_flow=np.zeros(n, np.float32), has_gefs=np.ones(n, bool),
        has_geosmart=np.ones(n, bool), window_start=pd.date_range("2024-01-01", periods=n),
        window_end=pd.date_range("2024-01-02", periods=n), bfi=np.zeros((n, 1), np.float32),
        geo_members=(levels + 10).astype(np.float32), geo_n=np.full(n, members), geo_origin=origin,
    )
    assert D.geosmart_representation(s) is s
    anomaly = D.geosmart_representation(s, "anomaly", "members")
    assert np.allclose(anomaly.geo_members, levels) and np.allclose(anomaly.dec[:, :, D.GEO_CH], levels.mean(1))
    mean = D.geosmart_representation(s, "level", "mean")
    assert mean.geo_members.shape == (n, 1, tokens) and (mean.geo_n == 1).all()
    assert np.allclose(mean.geo_members[:, 0], (levels + 10).mean(1))
    assert np.allclose(D.geosmart_representation(s, "anomaly", "mean").geo_members[:, 0], levels.mean(1))
    with pytest.raises(ValueError):
        D.geosmart_representation(s, "bogus")


def test_classification_metrics():
    truth = np.array([1, 1, 0, 0, 0, 1])
    pred = np.array([1, 0, 1, 0, 0, 1])
    m = classification_metrics(truth, pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["f1"] == pytest.approx(2 / 3)


def test_pinball_known_value():
    y = np.array([[1.0]])
    pred = np.array([[[0.0, 2.0]]])
    out = pinball(y, pred, [0.1, 0.9])
    np.testing.assert_allclose(out, [0.1, 0.1])


def test_end_to_end_full_size_config(tmp_path):
    """Default 14-day history / 14-day horizon on synthetic data: trains both variants and writes all artefacts."""
    from pipeline.train import run_training

    cfg = small_cfg(**{"paths.models_dir": tmp_path / "models", "paths.results_dir": tmp_path / "results"})
    cfg["paths"]["models_dir"], cfg["paths"]["results_dir"] = str(tmp_path / "models"), str(tmp_path / "results")
    hde, gefs, geo, _ = _synthetic(cfg, n_days=330)
    comp = run_training(cfg, hde, gefs, geo, run_id="t")
    assert cfg["derived"]["horizon_steps"] == 112
    for v in ("with_geosmart", "without_geosmart"):
        assert (tmp_path / "models/chilgrove/t" / v / "model.pt").exists()
        assert (tmp_path / "models/chilgrove/t" / v / "config.yaml").exists()
        assert (tmp_path / "results/chilgrove/t" / f"metrics_{v}.json").exists()
    assert (tmp_path / "results/chilgrove/summary.csv").exists()
    assert {"with_geosmart", "without_geosmart"} <= set(comp.index)
    assert {"test.skill_vs_climatology", "test.skill_vs_persistence"} <= set(comp.columns)


def test_model_description_matches_config():
    from pathlib import Path

    from pipeline.describe import config_hash

    doc = Path("MODEL.md").read_text()
    assert f"config-hash: {config_hash(load_config(CFG_PATH, 'chilgrove'))}" in doc, (
        "MODEL.md is out of date; run `python -m pipeline.describe chilgrove`"
    )


def test_geosmart_anchor_skips_missing_history_levels():
    issue = pd.Timestamp("2020-03-10 09:15")
    df = geosmart_frame(issue)
    last_hist = df.index[df["TimeSeriesID"] == 0][-1]
    df.loc[last_hist, "Level"] = np.nan
    fc = D.parse_geosmart_file(df, 0.9, 2, 0)
    assert fc.anchor_time == issue - pd.Timedelta(minutes=15)
    assert not np.isnan(fc.anchor_level)


def test_baselines_shapes_and_values():
    from pipeline.baselines import climatology, persistence

    y = np.tile(np.arange(10, dtype=float), (5, 1))
    c = climatology(y, 3, [0.1, 0.5, 0.9])
    assert c.shape == (3, 10, 3) and np.all(np.diff(c, axis=-1) >= 0)
    np.testing.assert_allclose(c[0, 0], np.quantile(np.arange(10), [0.1, 0.5, 0.9]))
    p = persistence(np.array([1.0, 2.0]), 4, 3)
    assert p.shape == (2, 4, 3) and (p[1] == 2.0).all()


def test_last_flow_is_value_at_origin():
    cfg = small_cfg(**{"data.history_days": 2, "data.horizon_days": 7, "data.decoder_days": 7})
    hde, gefs, geo, _ = _synthetic(cfg)
    hde["flow"] = np.arange(len(hde), dtype=float)
    s = D.build_samples(cfg, hde, gefs, geo)
    assert s.last_flow[0] == hde.index.get_loc(s.origins[0])


def test_weekly_metrics_cover_every_week():
    from pipeline.evaluate import evaluate_split, week_slices

    spd = 4
    assert list(week_slices(28 * spd, spd)) == ["week_1", "week_2", "week_3", "week_4"]
    y = np.zeros((3, 28 * spd))
    y[:, 7 * spd:] = 1.0  # flow is 0 in week 1 and 1 afterwards
    pred = np.stack([np.zeros_like(y)] * 3, axis=-1)  # predicts 0 everywhere
    m = evaluate_split(y, pred, [0.1, 0.5, 0.9], spd, threshold=0.5, prediction_quantile=0.5)
    assert set(m["by_week"]) == {"week_1", "week_2", "week_3", "week_4"}
    assert m["by_week"]["week_1"]["median_mae"] == 0.0 and m["by_week"]["week_2"]["median_mae"] == 1.0
    assert m["by_week"]["week_1"]["extreme_events"]["per_window"]["false_negative_rate"] is None  # no events in week 1
    assert m["by_week"]["week_3"]["extreme_events"]["per_window"]["false_negative_rate"] == 1.0


def test_horizon_must_be_whole_weeks():
    with pytest.raises(ValueError):
        small_cfg(**{"data.horizon_days": 10, "data.decoder_days": 10})


def test_four_week_horizon_propagates_to_sequences():
    cfg = small_cfg(**{"data.history_days": 2})
    dv = cfg["derived"]
    assert (dv["horizon_steps"], dv["n_decoder_tokens"], dv["n_output_tokens"], dv["horizon_fine_steps"]) == (112, 112, 112, 2688)
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    assert s.y.shape[1] == 112 and s.dec.shape[1] == 112
    assert (s.window_end - s.origins == pd.Timedelta(days=28)).all()


def test_geosmart_uses_persistent_members_when_ensemble_shrinks():
    issue = pd.Timestamp("2020-03-10 09:15")
    df = geosmart_frame(issue, n_members=10, n_days=28)
    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M")
    late = (df["TimeSeriesID"] > 6) & (dates >= issue.normalize() + pd.Timedelta(days=15))
    fc = D.parse_geosmart_file(df[~late], 0.5, 2, 0)
    assert len(fc.knot_times) == 28                     # full horizon kept, not just the 14 full-ensemble days
    ids = np.arange(1, 7)                               # the 6 members that persist, used on every day
    np.testing.assert_allclose(fc.mean[0], 50 + ids.mean())
    np.testing.assert_allclose(fc.std[0], ids.std(ddof=1))


def test_pool_quantiles_identical_members_and_spread():
    from pipeline.ensemble import pool_quantiles

    taus = [0.05, 0.25, 0.5, 0.75, 0.95]
    q = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])                     # (M=1, S=1, Q)
    same = pool_quantiles(np.repeat(q, 6, axis=0), taus, 200)
    np.testing.assert_allclose(same, q[0], atol=0.05)
    shifted = pool_quantiles(np.concatenate([q, q + 10.0]), taus, 200)
    assert shifted[0, 0] < 2 and shifted[0, -1] > 14    # the mixture spans both members
    assert 3.5 < shifted[0, 2] < 13.5 and np.all(np.diff(shifted[0]) >= 0)


def test_samples_carry_member_trajectories():
    cfg = small_cfg(**{"data.history_days": 2, "data.horizon_days": 7, "data.decoder_days": 7})
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    i = int(np.flatnonzero(s.has_geosmart)[0])
    n = s.geo_n[i]
    assert n == 10 and s.geo_members.shape[1] == 10
    # the placeholder channel is the member mean, and every member starts from the same observed anchor
    np.testing.assert_allclose(s.geo_members[i, :n].mean(axis=0), s.dec[i, :, D.GEO_CH], rtol=1e-4)
    assert (s.geo_n[~s.has_geosmart] == 0).all()


def test_training_draws_members_and_pooled_inference_runs():
    from pipeline.ensemble import draw_members

    idx = draw_members(np.array([1, 5, 44]), np.random.default_rng(0), k=200)
    assert (idx.max(axis=1) < np.array([1, 5, 44])).all() and idx[2].max() > 20


def test_boughton_filter_and_bfi():
    from pipeline.baseflow import boughton_filter, daily_baseflow

    q = np.array([1.0, 1.0, 5.0, 3.0, 1.5, 1.0, 1.0])
    bf = boughton_filter(q, 0.95, 0.5)
    assert (bf <= q + 1e-12).all() and bf[0] == 1.0 and bf[2] < q[2]       # baseflow never exceeds flow, peaks are runoff
    idx = pd.date_range("2020-01-01", periods=10, freq="D").delete([4, 5])  # a gap restarts the filter
    flow = pd.Series([1, 1, 4, 1, 2, 2, 2, 2.0], index=idx)
    out = daily_baseflow(flow, 0.95, 0.5)
    assert out["baseflow"].iloc[4] == 2.0 and ((out["bfi"] >= 0) & (out["bfi"] <= 1)).all()


def test_regime_metrics_split_by_bfi_and_week():
    from pipeline.evaluate import regime_metrics

    spd = 4
    y = np.ones((2, 14 * spd))
    pred = np.zeros((2, 14 * spd, 3))
    bfi = np.full(y.shape, 0.95)
    bfi[:, 7 * spd:] = 0.5                        # week 1 baseflow-dominated, week 2 runoff-dominated
    bfi[0, 0] = np.nan
    r = regime_metrics(y, pred, [0.1, 0.5, 0.9], spd, bfi, 0.9)
    assert r["baseflow_dominated"]["n_steps"] == 2 * 7 * spd - 1 and r["runoff_dominated"]["n_steps"] == 2 * 7 * spd
    assert r["baseflow_dominated"]["by_week"]["week_2"]["n_steps"] == 0
    assert r["runoff_dominated"]["by_week"]["week_2"]["median_mae"] == 1.0


def test_samples_carry_baseflow_index():
    cfg = small_cfg(**{"data.history_days": 2, "data.horizon_days": 7, "data.decoder_days": 7})
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    assert s.bfi.shape == s.y.shape and np.nanmin(s.bfi) >= 0 and np.nanmax(s.bfi) <= 1


def _mode_cfg(mode, tmp_path=None, **extra):
    return small_cfg(**{"data.target_mode": mode, "data.history_days": 2, "data.horizon_days": 7,
                        "data.decoder_days": 7, **extra})


def test_target_offsets_invert_exactly_in_every_mode():
    from pipeline.target import TARGET_MODES, seasonal_table, target_offset

    cfg = _mode_cfg("absolute")
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    q = cfg["model"]["quantiles"]
    table = seasonal_table(hde, s.window_end.max(), 360, q, 15)
    for mode in TARGET_MODES:
        off = target_offset(mode, s, table, q, 360)
        sc = D.Scalers.fit(s, off)
        enc, dec, y = sc.transform(s, [3, 4], off)
        back = sc.inverse_flow(np.repeat(y[..., None], 3, axis=-1), off)[..., 0]
        np.testing.assert_allclose(back, s.y, rtol=1e-3, atol=1e-3)
    off = target_offset("delta_log_last", s, table, q, 360)
    np.testing.assert_allclose(off[:, 0], np.log1p(s.anchor_flow), rtol=1e-6)
    assert np.allclose(target_offset("absolute", s, table, q, 360), 0)


def test_anchor_is_recent_mean_and_only_an_input_for_delta_mode():
    cfg = _mode_cfg("delta_log_last", **{"data.anchor_hours": 6})
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    i = 3
    i0 = hde.index.get_loc(s.origins[i])
    expected = hde["flow"].iloc[i0 - 23: i0 + 1].mean()
    assert abs(s.anchor_flow[i] - expected) < 1e-3
    np.testing.assert_allclose(s.dec[i, :, 5], np.log1p(s.anchor_flow[i]), rtol=1e-5)
    assert 5 in D.variant_dec_channels(True, True, True) and 5 not in D.variant_dec_channels(True, True)


def test_seasonal_climatology_uses_only_data_up_to_cutoff_and_is_seasonal():
    from pipeline.target import seasonal_climatology, seasonal_table

    idx = pd.date_range("2018-01-01", "2021-12-31 23:45", freq="15min")
    doy = idx.dayofyear.to_numpy()
    flow = 1.0 + np.cos(2 * np.pi * (doy - 20) / 365.25)          # winter high, summer low
    hde = pd.DataFrame({"flow": flow}, index=idx)
    q = [0.1, 0.5, 0.9]
    table = seasonal_table(hde, pd.Timestamp("2020-12-31"), 360, q, 15)
    assert table[20, 1] > 1.9 and table[200, 1] < 0.1
    future = hde.copy()
    future.loc["2021":, "flow"] = 99.0                             # data after the cutoff must not matter
    np.testing.assert_allclose(seasonal_table(future, pd.Timestamp("2020-12-31"), 360, q, 15), table)
    clim = seasonal_climatology(table, pd.DatetimeIndex(["2021-01-10 12:00"]), 8, 360)
    assert clim.shape == (1, 8, 3) and (clim[..., 1] > 1.5).all()


@pytest.mark.parametrize("mode", ["delta_log_last", "anomaly"])
def test_training_runs_in_each_target_mode(tmp_path, mode):
    from pipeline.train import run_training

    cfg = small_cfg(**{"data.target_mode": mode})
    cfg["paths"]["models_dir"], cfg["paths"]["results_dir"] = str(tmp_path / "m"), str(tmp_path / "r")
    cfg["training"]["epochs"] = 2
    hde, gefs, geo, _ = _synthetic(cfg, n_days=330)
    comp = run_training(cfg, hde, gefs, geo, run_id="t")
    assert {"test.skill_vs_seasonal_climatology", "test.skill_vs_persistence"} <= set(comp.columns)
    assert any(c.startswith("test.regime.baseflow_dominated.week_1") for c in comp.columns)


def test_delta_anomaly_offset_is_climatology_plus_origin_anomaly():
    from pipeline.target import seasonal_table, target_offset

    cfg = _mode_cfg("absolute")
    hde, gefs, geo, _ = _synthetic(cfg)
    s = D.build_samples(cfg, hde, gefs, geo)
    q = cfg["model"]["quantiles"]
    table = seasonal_table(hde, s.window_end.max(), 360, q, 15)
    an = target_offset("anomaly", s, table, q, 360)
    off = target_offset("delta_log_last_anomaly", s, table, q, 360)
    origin_clim = table[s.origins.dayofyear.to_numpy(), int(np.argmin(np.abs(np.asarray(q) - 0.5)))]
    shift = np.log1p(s.anchor_flow) - np.log1p(origin_clim)
    np.testing.assert_allclose(off - an, np.broadcast_to(shift[:, None], off.shape), rtol=1e-4, atol=1e-5)


def test_station_matching_helpers():
    import pandas as pd

    from processing.match_stations import classify_15min, match_geosmart, normalise

    assert normalise("Chilgrove House") == normalise("ChilgroveHouse")
    prefixes = pd.DataFrame({"geosmart_prefix": ["ChilgroveHouse", "BishopSutton"]})
    assert match_geosmart("Chilgrove House", prefixes, {}) == "ChilgroveHouse"
    assert match_geosmart("Bishops Sutton", prefixes, {normalise("Bishops Sutton"): "BishopSutton"}) == "BishopSutton"
    assert match_geosmart("Bishops Sutton", prefixes, {normalise("Bishops Sutton"): "Missing"}) is None
    assert match_geosmart("Unknown", prefixes, {}) is None

    st = pd.DataFrame({
        "stationGuid": ["f", "r", "g", "g_hourly"],
        "type": ["Station", "Station", "Groundwater|Station", "Groundwater|Station"],
        "measures.parameter": ["flow|flow", "rainfall|rainfall", "level|level", "level"],
        "measures.period": ["86400|900", "86400|900", "900|900", "3600"],
    })
    out = classify_15min(st).set_index("stationGuid")
    assert out.loc["f", "has_flow_15min"] and not out.loc["f", "has_rainfall_15min"]
    assert out.loc["r", "has_rainfall_15min"]
    assert out.loc["g", "has_groundwater_15min"] and not out.loc["g_hourly", "has_groundwater_15min"]


def test_one_flow_station_per_borehole():
    import pandas as pd

    from processing.match_stations import one_flow_station_per_borehole

    m = pd.DataFrame({
        "catchment_id": [1, 1, 1, 2],
        "flow_name": ["a", "b", "c", "d"],
        "groundwater_guid": ["g1", "g1", "g2", "g3"],
        "groundwater_distance_m": [500, 100, 50, 900],
    })
    out = one_flow_station_per_borehole(m)
    assert out["flow_name"].tolist() == ["c", "d"]


def test_geosmart_download_paths():
    from datetime import date
    from pathlib import Path

    from processing.download_geosmart import local_path, s3_uri

    d = date(2020, 1, 1)
    assert s3_uri("ChilgroveHouse", d) == (
        "s3://gwforecasting.geosmartinfo.co.uk/2020/01_January/01/ChilgroveHouse_headTimeSeries 2020-01-01.csv"
    )
    assert local_path("ChilgroveHouse", d, Path("data/geosmart")) == Path(
        "data/geosmart/chilgrovehouse/2020/01/ChilgroveHouse_headTimeSeries 2020-01-01.csv"
    )


def test_geosmart_store_consolidates_and_matches_raw_parsing(tmp_path):
    from pipeline import geosmart_store as S
    folder = tmp_path / "bh"
    raw = {}
    for day in ("2020-03-10", "2020-03-11"):
        df = geosmart_frame(pd.Timestamp(f"{day} 09:15"))
        raw[day] = df
        f = folder / day[:4] / day[5:7] / f"Bh_headTimeSeries {day}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        df.assign(**{"Forecast On": day}).to_csv(f, index=False)
    assert S.consolidate(folder, "Bh", delete=True) == 2
    assert not list(folder.rglob("*.csv")) and S.store_path(folder, "Bh").exists()
    assert S.stored_issue_dates(folder, "Bh") == set(raw)
    cfg = dict(min_member_fraction=0.9, min_members=2, availability_lag_hours=0)
    loaded = {fc.issue_time: fc for fc in D.load_geosmart(folder, "Bh", cfg)}
    assert len(loaded) == 2
    for day, df in raw.items():
        want = D.parse_geosmart_file(df, 0.9, 2, 0)
        got = loaded[want.issue_time]
        assert got.anchor_time == want.anchor_time and got.anchor_level == want.anchor_level
        np.testing.assert_array_equal(got.mean, want.mean)
        np.testing.assert_array_equal(got.members, want.members)


def test_geosmart_store_keeps_day_without_history(tmp_path):
    from pipeline import geosmart_store as S
    f = tmp_path / "2024" / "10" / "Bh_headTimeSeries 2024-10-04.csv"
    f.parent.mkdir(parents=True)
    pd.DataFrame({"TimeSeriesID": [], "Date": [], "Level": []}).to_csv(f, index=False)
    S.consolidate(tmp_path, "Bh", delete=True)
    assert S.stored_issue_dates(tmp_path, "Bh") == {"2024-10-04"}


def test_target_series_components():
    from pipeline.baseflow import daily_baseflow, target_series

    idx = pd.date_range("2020-01-01", periods=96 * 12, freq="15min")
    rng = np.random.default_rng(0)
    flow = pd.Series(1 + np.abs(rng.normal(size=len(idx))) + (idx.day % 4 == 0) * 3.0, index=idx)
    flow.iloc[200:210] = np.nan
    base = target_series(flow, "baseflow", 0.95, 0.5)
    run = target_series(flow, "runoff", 0.95, 0.5)
    total = target_series(flow, "total_flow", 0.95, 0.5)
    assert total["target"].equals(flow)
    ok = flow.notna()
    daily_mean = flow.resample("D").mean().reindex(idx.normalize()).to_numpy()
    assert (base["target"][ok].to_numpy() <= daily_mean[ok] + 1e-9).all()
    assert (run["target"][ok] >= 0).all()
    assert base["target"][~ok].isna().all() and run["target"][~ok].isna().all()
    daily = daily_baseflow(flow, 0.95, 0.5)["baseflow"]
    day3 = idx.normalize() == pd.Timestamp("2020-01-03")
    assert np.allclose(base["target"][day3 & ok], daily[pd.Timestamp("2020-01-03")])
    # origin-safe form uses the previous day's baseflow, never the current day's
    assert np.allclose(base["causal"][day3 & ok], daily[pd.Timestamp("2020-01-02")])
    with pytest.raises(ValueError):
        target_series(flow, "nonsense", 0.95, 0.5)
