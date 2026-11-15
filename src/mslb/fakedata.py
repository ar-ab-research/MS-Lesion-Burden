"""Fake clinical and FC tables, for trying the pipeline without real data.

The numbers are random. Two associations are planted on purpose so a test run has
something to find, and so the tests can check that it is found:
  - nbss_total rises with lesion burden
  - the first connection's full-state value falls with lesion burden
Everything else is noise. When no burden is passed in, a random one stands in, so
the planted effects only line up with the imaging if the real burden is supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLANTED_CONNECTION = "net01__net02"
# every fake clinical table carries this column, and the summary says so loudly when it sees it
FAKE_COLUMN = "is_fake"


def connection_names(n_networks: int = 4, n_seed_targets: int = 4) -> list[str]:
    """Generic placeholder labels: network pairs, then seed-to-region values."""
    nets = [f"net{i + 1:02d}" for i in range(n_networks)]
    pairs = [f"{a}__{b}" for i, a in enumerate(nets) for b in nets[i + 1:]]
    seeds = [f"seed01__roi{i + 1:02d}" for i in range(n_seed_targets)]
    return pairs + seeds


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order + 0.5) / len(values)


def make_fake(subject_ids, seed: int = 0, burden=None, strength: float = 0.8,
              with_treatment: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clinical, fc) in the template layouts.

    burden: optional mapping of subject ID to any lesion-burden number. Only its
    ranks are used. strength is the share of the planted variables that follows
    burden, between 0 (none) and 1 (all).
    """
    ids = [str(s) for s in subject_ids]
    n = len(ids)
    if n < 2 or len(set(ids)) != n:
        raise ValueError("need at least two distinct subject IDs")
    rng = np.random.default_rng(seed)
    if burden is None:
        latent = rng.lognormal(mean=1.0, sigma=1.0, size=n)
    else:
        latent = np.array([float(burden[s]) for s in ids])
        # ties (several zero-lesion subjects) are broken at random so ranks stay spread
        latent = latent + rng.uniform(0, 1e-9, size=n)
    z = (_rank01(latent) - 0.5) * np.sqrt(12)  # unit variance, follows burden rank

    def mix(noise_scale: float = 1.0) -> np.ndarray:
        return strength * z + np.sqrt(1 - strength**2) * rng.normal(size=n) * noise_scale

    clinical = pd.DataFrame({"subject_id": ids})
    clinical[FAKE_COLUMN] = 1
    clinical["do_present"] = rng.integers(0, 2, size=n)
    # keep both groups populated even in a tiny cohort
    clinical.loc[clinical.index[0], "do_present"] = 1
    clinical.loc[clinical.index[1], "do_present"] = 0
    clinical["nbss_total"] = np.clip(np.round(30 + 9 * mix()), 0, 74).astype(int)
    clinical["diary_voids_per_day"] = np.round(np.clip(rng.normal(10, 3, n), 3, None), 1)
    clinical["diary_urgency_per_day"] = np.round(np.clip(rng.normal(5, 2.5, n), 0, None), 1)
    clinical["diary_incontinence_per_day"] = np.round(np.clip(rng.normal(1.5, 1.5, n), 0, None), 1)
    if with_treatment:
        arm = np.where(rng.random(n) < 2 / 3, "active", "sham")
        arm[:2] = ["active", "sham"]
        clinical["arm"] = arm
        drop = np.where(arm == "active", rng.normal(7, 6, n), rng.normal(3, 5, n))
        clinical["nbss_total_post"] = np.clip(
            np.round(clinical["nbss_total"] - drop), 0, 74).astype(int)

    rows = []
    for conn in connection_names():
        shared = rng.normal(0, 0.15, n)  # a subject's two states resemble each other
        for state in ("full", "empty"):
            values = 0.2 + shared + rng.normal(0, 0.12, n)
            if conn == PLANTED_CONNECTION and state == "full":
                values = 0.2 - 0.2 * mix()
            rows.append(pd.DataFrame({
                "subject_id": ids, "state": state, "connection": conn,
                "fc_z": np.round(values, 4),
            }))
    fc = pd.concat(rows, ignore_index=True)
    return clinical, fc


def default_ids(n: int) -> list[str]:
    return [f"sub-{i + 1:03d}" for i in range(n)]
