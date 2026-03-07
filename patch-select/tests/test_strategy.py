import numpy as np

from src.preprocess.strategy import infer_sparse_canvas_mode, resolve_stain_vectors, retry_acceptance


def test_infer_sparse_canvas_mode_by_suffix_and_canvas_stats():
    cfg = {"sparse_canvas_black_frac_min": 0.75, "sparse_canvas_bbox_frac_max": 0.15}

    assert infer_sparse_canvas_mode(".mrxs", black_canvas_frac=0.10, non_black_bbox_frac=0.90, cfg=cfg) is True
    assert infer_sparse_canvas_mode(".svs", black_canvas_frac=0.80, non_black_bbox_frac=0.50, cfg=cfg) is True
    assert infer_sparse_canvas_mode(".ndpi", black_canvas_frac=0.20, non_black_bbox_frac=0.10, cfg=cfg) is True
    assert infer_sparse_canvas_mode(".svs", black_canvas_frac=0.20, non_black_bbox_frac=0.40, cfg=cfg) is False


def test_retry_acceptance_rejects_white_leak_and_accepts_clean_gain():
    cfg = {
        "low_tissue_retry_min_gain": 0.0015,
        "low_tissue_retry_min_added_px": 3000,
        "low_tissue_retry_min_nonwhite_added_frac": 0.85,
        "low_tissue_retry_max_white_leak_increase": 0.01,
    }

    rejected = retry_acceptance(
        retry_gain=0.0020,
        retry_added_px=5000,
        retry_added_nonwhite_frac=0.90,
        retry_white_leak_delta=0.05,
        new_status_effective="ok",
        cfg=cfg,
    )
    assert rejected is False

    accepted = retry_acceptance(
        retry_gain=0.0020,
        retry_added_px=5000,
        retry_added_nonwhite_frac=0.90,
        retry_white_leak_delta=0.005,
        new_status_effective="ok",
        cfg=cfg,
    )
    assert accepted is True


def test_resolve_stain_vectors_estimates_from_thumbnail_signal():
    h_vec = np.array([0.651, 0.701, 0.290], dtype=np.float32)
    h_vec = h_vec / np.linalg.norm(h_vec)
    e_vec = np.array([0.216, 0.801, 0.558], dtype=np.float32)
    e_vec = e_vec / np.linalg.norm(e_vec)

    rng = np.random.default_rng(123)
    h = np.zeros((256, 256), dtype=np.float32)
    e = np.zeros((256, 256), dtype=np.float32)
    h[:, :128] = rng.uniform(0.40, 1.10, size=(256, 128))
    e[:, :128] = rng.uniform(0.05, 0.35, size=(256, 128))
    h[:, 128:] = rng.uniform(0.05, 0.35, size=(256, 128))
    e[:, 128:] = rng.uniform(0.40, 1.10, size=(256, 128))
    od = (h[..., None] * h_vec[None, None, :]) + (e[..., None] * e_vec[None, None, :]) + 0.02
    rgb = np.clip(np.exp(-od) * 255.0, 0, 255).astype(np.uint8)

    out = resolve_stain_vectors(rgb, cfg={"dynamic_stain_vectors_enabled": True, "stain_estimation_min_pixels": 500})
    assert out["stain_vector_source"] == "estimated"

    h_est = np.asarray(out["stain_vector_h"], dtype=np.float32)
    h_est = h_est / np.linalg.norm(h_est)
    e_est = np.asarray(out["stain_vector_e"], dtype=np.float32)
    e_est = e_est / np.linalg.norm(e_est)
    assert float(np.dot(h_est, h_vec)) > 0.85
    assert float(np.dot(e_est, e_vec)) > 0.85


def test_resolve_stain_vectors_fallback_on_low_signal():
    rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
    out = resolve_stain_vectors(rgb, cfg={"dynamic_stain_vectors_enabled": True, "stain_estimation_min_pixels": 200})
    assert out["stain_vector_source"] == "fallback"
    assert out["stain_estimation_reason"] == "too_few_pixels"


def test_resolve_stain_vectors_fallback_on_unstable_he_geometry():
    # Strong signal, but forced strict geometry gate should trigger fallback.
    rng = np.random.default_rng(7)
    rgb = np.full((192, 192, 3), 250, dtype=np.uint8)
    rgb[:, :, 0] = rng.integers(110, 190, size=(192, 192), dtype=np.uint8)
    rgb[:, :, 1] = rng.integers(80, 170, size=(192, 192), dtype=np.uint8)
    rgb[:, :, 2] = rng.integers(100, 190, size=(192, 192), dtype=np.uint8)
    out = resolve_stain_vectors(
        rgb,
        cfg={
            "dynamic_stain_vectors_enabled": True,
            "stain_estimation_min_pixels": 200,
            "stain_estimation_max_he_dot": 0.10,
        },
    )
    assert out["stain_vector_source"] == "fallback"
    assert out["stain_estimation_reason"] == "he_unstable"
