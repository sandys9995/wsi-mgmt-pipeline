import numpy as np
import pandas as pd

from src.select.sampling import quota_select


def test_quota_select_allows_under_target_with_type_d_cap():
    df = pd.DataFrame(
        {
            "cell_rich_p": np.linspace(1.0, 0.1, 10),
            "type": ["A", "A", "B", "B", "C", "D", "D", "D", "D", "D"],
        }
    )
    cfg = {
        "scoring": {
            "quotas": {
                "typeA_frac": 0.5,
                "typeB_frac": 0.3,
                "typeC_frac": 0.2,
                "typeD_frac": 0.0,
            },
            "blood_cap_frac": 0.2,
            "allow_under_target": True,
            "max_typeD_frac": 0.1,
        }
    }
    target = 10
    sel = quota_select(df, target=target, cfg=cfg)
    sel_df = df.iloc[sel]

    assert len(sel) < target
    assert int((sel_df["type"] == "D").sum()) <= 1
