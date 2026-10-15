"""End-to-end: a small run, its CSV, manifest and analysis outputs."""
import json

import pandas as pd
import pytest

from src.metrics import CSV_FIELDS
from src.run_experiment import main

SAMPLES = 3


def run(out_root, analysis=True):
    args = ["--samples", str(SAMPLES), "--out-root", str(out_root)]
    assert main(args + ([] if analysis else ["--no-analysis"])) == 0
    (run_dir,) = list(out_root.iterdir())
    return run_dir


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return run(tmp_path_factory.mktemp("run"))


def test_rows_complete(run_dir):
    df = pd.read_csv(run_dir / "raw" / "results.csv")
    assert list(df.columns) == CSV_FIELDS
    assert len(df) == 6 * SAMPLES * 4 * 3
    assert not df[["regime", "sample_id", "operator", "quantizer"]].duplicated().any()
    assert (df.ref_finite == 1).all() and (df.quant_finite == 1).all()


def test_inapplicable_fields_are_empty(run_dir):
    df = pd.read_csv(run_dir / "raw" / "results.csv")
    assert df.loc[df.operator == "sum", "sqnr_db"].isna().all()
    assert df.loc[df.operator != "softmax", "softmax_kl"].isna().all()
    assert df.loc[df.quantizer != "Q2", "group_size"].isna().all()


def test_manifest(run_dir):
    m = json.loads((run_dir / "manifest.json").read_text())
    assert m["status"] == "ok" and m["samples_per_regime"] == SAMPLES
    assert m["machine"]["torch_num_threads"] == 1 and m["machine"]["torch_cpu_capability"]
    c = m["counts"]
    assert c["rows"] == c["expected_rows"] and c["missing_rows"] == c["non_finite_rows"] == 0


def test_analysis_outputs(run_dir):
    for rel in ("summary/summary.csv", "summary/summary.md", "plots/F1_inputs.png",
                "plots/F2_error_heatmap.png", "plots/F3_amplification.png", "run.log"):
        assert (run_dir / rel).stat().st_size > 0, rel
    s = pd.read_csv(run_dir / "summary" / "summary.csv")
    assert len(s) == 4 * 6 * 3 and (s.n_samples == SAMPLES).all()
    md = (run_dir / "summary" / "summary.md").read_text(encoding="utf-8")
    for heading in ("Loss of precision", "more sensitive than elementwise", "best scheme depend"):
        assert heading in md


def test_rerun_is_bit_identical(run_dir, tmp_path):
    again = run(tmp_path, analysis=False)
    a = pd.read_csv(run_dir / "raw" / "results.csv", dtype=str).drop(columns="run_id")
    b = pd.read_csv(again / "raw" / "results.csv", dtype=str).drop(columns="run_id")
    pd.testing.assert_frame_equal(a, b)
