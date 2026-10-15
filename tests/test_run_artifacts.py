"""Checks on run outputs (plan §8.1: scale freeze, completeness).

These need Gate 1 and Gate 2 outputs, so they skip until those artifacts exist.
"""
import json
from collections import Counter

import pandas as pd
import pytest

from src.config import REPORTED_SPLITS, ROOT, expected_row_keys, file_checksum, stratum_key

ARTIFACTS = ROOT / "artifacts"
SCALES = ARTIFACTS / "calibration" / "scales.pt"


def run_manifests():
    return [(p.parent, json.loads(p.read_text())) for p in sorted(ARTIFACTS.glob("*/manifest.json"))]


def raw_csvs():
    return [p for split in REPORTED_SPLITS for ext in ("csv", "csv.gz")
            for p in sorted(ARTIFACTS.glob(f"*/raw/{split}.{ext}"))]


def test_static_scale_checksum_frozen():
    if not SCALES.exists():
        pytest.skip("static scales are frozen at Gate 1")
    reported = [m for _, m in run_manifests() if set(m.get("splits", [])) & set(REPORTED_SPLITS)]
    if not reported:
        pytest.skip("no development or held-out run yet")
    expected = file_checksum(SCALES)
    assert {m["static_scale_checksum"] for m in reported} == {expected}


@pytest.mark.parametrize("path", raw_csvs() or [None])
def test_rows_complete_and_unique(cfg, path):
    if path is None:
        pytest.skip("no raw CSVs yet (Gate 2)")
    df = pd.read_csv(path, usecols=["split", "regime", "sweep_id", "sweep_value", "sample_id",
                                    "operator", "quantizer", "calib_policy"])
    keys = [
        (r.split, stratum_key(r.regime, None if pd.isna(r.sweep_id) else r.sweep_id,
                              None if pd.isna(r.sweep_value) else r.sweep_value),
         int(r.sample_id), r.operator, r.quantizer, r.calib_policy)
        for r in df.itertuples(index=False)
    ]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert not dupes, f"{len(dupes)} duplicated rows, e.g. {dupes[:3]}"
    split = df["split"].iloc[0]
    present = {k[1] for k in keys}
    expected = {k for k in expected_row_keys(cfg, split) if k[1] in present}
    assert set(cfg["regimes"]) <= present, "core regimes missing"
    assert set(keys) == expected
