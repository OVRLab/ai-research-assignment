import json

import pytest
import torch
from behavior import validate_pair_settings
from common import ROOT
from edit import preserve_norm_edit
from report import paired_counts


def test_zero_intervention_preserves_original_exactly():
    w = torch.randn(8, 16, dtype=torch.bfloat16)
    assert torch.equal(w, preserve_norm_edit(w, torch.randn(8), 0))


def test_rectangular_matrix_preserves_output_row_norms():
    torch.manual_seed(42)
    w = torch.randn(8, 16)
    result = preserve_norm_edit(w, torch.randn(8), 0.5)
    torch.testing.assert_close(w.norm(dim=1), result.norm(dim=1))
    assert not torch.equal(w, result)
    assert result.shape == w.shape


def test_bf16_rounding_is_bounded():
    torch.manual_seed(42)
    w = torch.randn(16, 32, dtype=torch.bfloat16)
    result = preserve_norm_edit(w, torch.randn(16), 0.5)
    error = (w.float().norm(dim=1) - result.float().norm(dim=1)).abs()
    assert (error / w.float().norm(dim=1)).max() < 0.005
    assert result.dtype == w.dtype


def test_zero_rows_remain_finite():
    w = torch.zeros(8, 16)
    assert torch.equal(w, preserve_norm_edit(w, torch.randn(8), 0.5))


@pytest.mark.parametrize("direction", [torch.zeros(8), torch.full((8,), float("nan"))])
def test_invalid_directions_are_rejected(direction):
    with pytest.raises(ValueError):
        preserve_norm_edit(torch.randn(8, 16), direction, 0.5)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        preserve_norm_edit(torch.randn(8, 16), torch.randn(16), 0.5)


def test_collapsed_nonzero_row_is_rejected():
    with pytest.raises(ValueError):
        preserve_norm_edit(torch.eye(2), torch.tensor([1.0, 0.0]), 1)


def test_paired_report_exposes_offsetting_regressions():
    before = [{"id": "a", "correct": True}, {"id": "b", "correct": False}]
    after = [{"id": "b", "correct": True}, {"id": "a", "correct": False}]
    assert paired_counts(before, after) == (1, 1, 0)


def test_mismatched_samples_are_rejected():
    with pytest.raises(ValueError):
        paired_counts([{"id": "a", "correct": True}], [{"id": "b", "correct": True}])


def test_duplicate_sample_ids_are_rejected():
    with pytest.raises(ValueError):
        paired_counts(
            [{"id": "a", "correct": True}, {"id": "a", "correct": False}],
            [{"id": "a", "correct": True}],
        )


def test_development_and_test_prompts_are_disjoint():
    dev = json.loads((ROOT / "data/dev.json").read_text())
    test = json.loads((ROOT / "data/test.json").read_text())
    calibration = (ROOT / "data/calibration.txt").read_text().splitlines()
    assert len(test) == 20
    assert sum(row["requests_detail"] for row in test) == 5
    prompts = [row["prompt"] for row in dev + test] + calibration
    assert len(prompts) == len(set(prompts))


def test_different_runtime_templates_are_rejected():
    with pytest.raises(ValueError, match="template"):
        validate_pair_settings({"template": "original"}, {"template": "changed"})


def test_different_quantization_is_rejected():
    with pytest.raises(ValueError, match="quantization_level"):
        validate_pair_settings(
            {"details": {"quantization_level": "F16"}},
            {"details": {"quantization_level": "Q4_0"}},
        )
