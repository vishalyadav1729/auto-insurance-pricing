"""Policy-level train/validation/test split.

The split is performed once, on unique policy IDs (IDpol), and must be
applied to any downstream table (frequency or severity) by joining on
IDpol - never by independently sampling the frequency and severity tables.
If the severity table were split on its own, some of a single policy's
claims could end up in train while others end up in test, leaking
information about that exact policy between partitions.

The split is a reproducible random split, stratified by whether the policy
had any claim (ClaimNb > 0). freMTPL2 does not expose per-policy dates at a
resolution useful for a genuine chronological holdout, so a temporal split
is not attempted here - this is a documented limitation, not something
silently assumed away.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_policies(
    freq: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Assign every policy in `freq` to train/validation/test.

    Stratifies by has_claim (ClaimNb > 0) so each split ends up with a
    similar claim rate - important because claims are rare (~5% of
    policies, per Phase 3), and a plain random split could by chance put a
    meaningfully different claim rate into one partition purely by luck.

    Returns a DataFrame with columns [IDpol, split], split one of
    "train"/"validation"/"test". Every IDpol in `freq` appears exactly once.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    has_claim = (freq["ClaimNb"] > 0).astype(int)

    train_ids, holdout_ids, _, holdout_strat = train_test_split(
        freq["IDpol"],
        has_claim,
        train_size=train_size,
        stratify=has_claim,
        random_state=random_state,
    )

    # Split the holdout into validation/test, preserving the requested
    # relative proportions (e.g. 0.15/0.15 of the total -> 50/50 of holdout).
    relative_val_size = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        holdout_ids,
        train_size=relative_val_size,
        stratify=holdout_strat,
        random_state=random_state,
    )

    return pd.concat(
        [
            pd.DataFrame({"IDpol": train_ids.to_numpy(), "split": "train"}),
            pd.DataFrame({"IDpol": val_ids.to_numpy(), "split": "validation"}),
            pd.DataFrame({"IDpol": test_ids.to_numpy(), "split": "test"}),
        ],
        ignore_index=True,
    )
