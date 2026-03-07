import numpy as np

from src.select.pipeline import _resolve_slide_diag


def test_resolve_slide_diag_prefers_effective_status_when_present():
    d = _resolve_slide_diag(
        {
            "mask_status": "low_tissue",
            "mask_status_effective": "ok",
            "status_basis": "effective",
            "mask_frac": 0.02,
            "mask_frac_effective": 0.20,
            "fallback_used": True,
            "sparse_canvas_mode": True,
            "profile_used": "sparse_canvas_pure_v1",
        }
    )
    assert d["mask_status"] == "ok"
    assert d["mask_status_global"] == "low_tissue"
    assert d["mask_status_effective"] == "ok"
    assert d["status_basis"] == "effective"


def test_resolve_slide_diag_falls_back_to_global_when_effective_missing():
    d = _resolve_slide_diag(
        {
            "mask_status": "ok",
            "mask_status_effective": np.nan,
            "mask_frac": 0.3,
            "fallback_used": False,
        }
    )
    assert d["mask_status"] == "ok"
    assert d["mask_status_global"] == "ok"
    assert d["mask_status_effective"] == "ok"
