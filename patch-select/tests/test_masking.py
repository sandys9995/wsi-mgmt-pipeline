import numpy as np

from src.preprocess.masking import build_tissue_mask, select_mask_with_fallback


def test_adaptive_mask_captures_pale_tissue():
    img = np.full((256, 256, 3), 245, dtype=np.uint8)
    # Pale tissue island (low saturation but non-white OD).
    img[64:192, 64:192, 0] = 220
    img[64:192, 64:192, 1] = 205
    img[64:192, 64:192, 2] = 220

    mask, stats, _ = build_tissue_mask(
        img,
        cfg={
            "expected_frac_min": 0.05,
            "expected_frac_max": 0.70,
            "bg_white_gray_min": 255,
            "white_void_gray_min": 255,
            "pale_void_gray_min": 255,
            "edge_strip_px": 0,
            "edge_guard_px": 0,
            "border_component_margin_px": 0,
            "final_edge_crop_px": 0,
            "score_thr": 0.08,
            "od_q": 25,
            "min_component_area_ratio": 0.00001,
        },
    )
    frac_inside = float(mask[64:192, 64:192].mean())
    assert frac_inside > 0.60
    assert stats["mask_frac"] > 0.10


def test_pen_like_pixels_removed_from_mask():
    img = np.full((256, 256, 3), 240, dtype=np.uint8)
    img[40:220, 40:220] = np.array([200, 140, 170], dtype=np.uint8)  # tissue region
    # Magenta pen stripe
    img[120:132, 20:236] = np.array([230, 30, 210], dtype=np.uint8)

    mask, _, _ = build_tissue_mask(img, cfg=None)
    stripe_frac = float(mask[120:132, 20:236].mean())
    tissue_frac = float(mask[60:180, 60:180].mean())
    assert tissue_frac > 0.70
    assert stripe_frac < 0.30


def test_fallback_triggers_for_extreme_low_coverage():
    img = np.full((256, 256, 3), 250, dtype=np.uint8)
    img[96:160, 96:160] = np.array([236, 228, 236], dtype=np.uint8)  # very pale region

    _, stats, _ = select_mask_with_fallback(
        img,
        cfg={
            "expected_frac_min": 0.15,
            "expected_frac_max": 0.60,
            "low_frac_trigger": 0.90,
            "high_frac_trigger": 0.70,
        },
    )
    assert stats["fallback_attempted"] is True
    assert stats["mask_status"] in {"ok", "low_tissue", "high_tissue"}


def test_sparse_canvas_uses_effective_status():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Sparse non-black region with a smaller tissue island.
    # This keeps global mask fraction low while effective fraction remains usable.
    img[40:216, 40:216] = np.array([240, 240, 240], dtype=np.uint8)
    img[96:160, 96:160] = np.array([170, 140, 150], dtype=np.uint8)

    mask, stats, _ = select_mask_with_fallback(
        img,
        cfg={
            "sparse_canvas_mode": True,
            "expected_frac_min": 0.12,
            "expected_frac_max": 0.62,
            "expected_frac_min_effective": 0.08,
            "expected_frac_max_effective": 0.80,
            "edge_strip_px": 0,
            "edge_guard_px": 0,
            "border_component_margin_px": 0,
            "final_edge_crop_px": 0,
            "score_thr": 0.0,
            "fallback_score_thr": 0.0,
            "od_q": 15,
            "fallback_od_q": 10,
            "min_component_area_ratio": 0.00001,
        },
    )
    assert int(mask.sum()) > 0
    assert stats["mask_frac"] < 0.12
    assert stats["mask_frac_effective"] > 0.08
    assert stats["mask_status"] == "low_tissue"
    assert stats["mask_status_effective"] == "ok"
    assert stats["status_basis"] == "effective"
