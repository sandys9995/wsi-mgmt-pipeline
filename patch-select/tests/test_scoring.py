import numpy as np

from src.select.scoring import compute_scores_and_types


def test_blood_score_not_degenerate_for_blood_like_patch():
    tiles = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    # Blood-like red patch
    tiles[0, :, :] = np.array([200, 40, 40], dtype=np.uint8)
    # Cell-rich purple patch
    tiles[1, :, :] = np.array([120, 80, 150], dtype=np.uint8)
    # Pale background-like patch
    tiles[2, :, :] = np.array([235, 225, 235], dtype=np.uint8)

    tf = np.array([1.0, 1.0, 0.2], dtype=np.float32)
    fs = np.array([20.0, 120.0, 5.0], dtype=np.float32)
    cfg = {"scoring": {"cell_rich_weights": {}}}

    df = compute_scores_and_types(tiles, tf=tf, fs=fs, cfg=cfg)
    assert "blood_score" in df.columns
    assert float(df["blood_score"].max()) > 0.0
    assert float(df["blood_score"].std()) > 0.0
