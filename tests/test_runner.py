import json

import pandas as pd
import pytest

from src.config import ROOT
from src.metrics import CSV_FIELDS
from src.run_experiment import main

SAMPLES = 2


def run_pilot(out_root, plots=False):
    args = ["--gate", "0", "--samples", str(SAMPLES), "--out-root", str(out_root)]
    assert main(args + ([] if plots else ["--no-plots"])) == 0
    (run_dir,) = list(out_root.iterdir())
    return run_dir


@pytest.fixture(scope="module")
def pilot_dir(tmp_path_factory):
    return run_pilot(tmp_path_factory.mktemp("pilot"), plots=True)


def test_pilot_rows_complete(pilot_dir):
    df = pd.read_csv(pilot_dir / "raw" / "pilot.csv")
    assert list(df.columns) == CSV_FIELDS
    assert len(df) == 6 * SAMPLES * 4 * 3 * 2
    keys = df[["regime", "sample_id", "operator", "quantizer", "calib_policy"]]
    assert not keys.duplicated().any()
    assert (df.ref_finite == 1).all() and (df.quant_finite == 1).all()


def test_static_rows_use_pilot_scales(pilot_dir):
    df = pd.read_csv(pilot_dir / "raw" / "pilot.csv")
    d1 = df[(df.regime == "D1") & (df.operator == "sum")]
    assert (d1[d1.calib_policy == "static"].clip_fraction > 0).all()   # ±20 outliers saturate D0 scales
    assert (d1[d1.calib_policy == "dynamic"].clip_fraction == 0).all()


def test_inapplicable_fields_are_empty(pilot_dir):
    df = pd.read_csv(pilot_dir / "raw" / "pilot.csv")
    assert df.loc[df.operator == "sum", "sqnr_db"].isna().all()
    assert df.loc[df.operator != "softmax", "softmax_kl"].isna().all()
    assert df.loc[df.quantizer != "Q2", "group_size"].isna().all()
    assert df.sweep_id.isna().all()


def test_manifest(pilot_dir):
    m = json.loads((pilot_dir / "manifest.json").read_text())
    assert m["status"] == "ok" and m["splits"] == ["pilot"]
    assert m["rounding_half_even"] is True
    assert m["machine"]["torch_num_threads"] == 1
    c = m["counts"]["pilot"]
    assert c["rows"] == c["expected_rows"] and c["missing_rows"] == c["non_finite_rows"] == 0
    assert m["static_scale_checksum"] is None   # pilot scales are throwaway


def test_sanity_outputs(pilot_dir):
    for rel in ("plots/pilot_histograms.png", "summary/regime_stats.csv",
                "summary/pilot_medians.csv", "summary/pilot_predictions_check.md", "run.log"):
        assert (pilot_dir / rel).stat().st_size > 0, rel


def test_pilot_never_writes_frozen_scales(pilot_dir):
    assert not list(pilot_dir.rglob("scales.pt"))
    assert not (ROOT / "artifacts" / "calibration" / "scales.pt").exists()


def test_rerun_is_bit_identical(pilot_dir, tmp_path):
    again = run_pilot(tmp_path)
    a = pd.read_csv(pilot_dir / "raw" / "pilot.csv", dtype=str).drop(columns="run_id")
    b = pd.read_csv(again / "raw" / "pilot.csv", dtype=str).drop(columns="run_id")
    pd.testing.assert_frame_equal(a, b)


def test_unimplemented_gate_fails_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        main(["--gate", "2", "--out-root", str(tmp_path)])
