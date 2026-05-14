"""Generate a sign-flipped Lalonde NSW dataset for the parrot diagnostic.

Loads the canonical Dehejia-Wahba NSW sample via ``causaldata`` and, for
every TREATED row (``treat == 1``), multiplies the 1978 earnings
(``re78``) outcome by ``-1``. The covariates and the treatment column
are left untouched.

Why this works as a parrot test
-------------------------------
The Lalonde NSW dataset is one of the most-memorized real datasets in
the causal-inference literature: every LLM trained on textbooks,
DoWhy/EconML docs, and arXiv has seen "training raises 1978 earnings"
phrased dozens of ways. A parroting LLM will, regardless of what the
csv actually contains, confidently propose hypotheses framed as
"training increases earnings" with an expected positive ATE.

By flipping the sign of ``re78`` on treated rows only, the *true*
average treatment effect on the persisted file is now decisively
NEGATIVE. A reasoning model that actually inspects the data (or at
least frames hypotheses neutrally) will land on the right sign after
estimation. A parrot will not.

Usage
-----
    python scripts/parrot_signflip_lalonde.py --out artifacts/lalonde_signflipped.csv

The default output path is ``artifacts/lalonde_signflipped.csv``
relative to the current working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_signflipped(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``re78`` negated on treated rows."""
    if "treat" not in df.columns or "re78" not in df.columns:
        raise ValueError(
            "Expected Lalonde NSW frame with 'treat' and 're78' columns; "
            f"got {list(df.columns)!r}"
        )
    out = df.copy()
    treated_mask = out["treat"] == 1
    out.loc[treated_mask, "re78"] = -1.0 * out.loc[treated_mask, "re78"]
    return out


def load_nsw() -> pd.DataFrame:
    """Load the Dehejia-Wahba NSW frame via ``causaldata``."""
    try:
        import causaldata  # type: ignore
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "causaldata is not installed. Install with: "
            "pip install 'causalrag[real-data]'"
        ) from e
    ds = causaldata.nsw_mixtape.load_pandas().data
    return ds.drop(columns=["data_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/lalonde_signflipped.csv"),
        help="Destination CSV path (default: artifacts/lalonde_signflipped.csv).",
    )
    args = parser.parse_args()

    df = load_nsw()
    flipped = build_signflipped(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flipped.to_csv(args.out, index=False)

    n_treated = int((df["treat"] == 1).sum())
    print(
        f"Wrote {len(flipped)} rows ({n_treated} treated) to {args.out}.\n"
        f"  treated.re78.mean  (orig)   = {df.loc[df.treat == 1, 're78'].mean():.2f}\n"
        f"  treated.re78.mean  (flipped) = {flipped.loc[flipped.treat == 1, 're78'].mean():.2f}\n"
        f"  control.re78.mean  (unchanged) = {flipped.loc[flipped.treat == 0, 're78'].mean():.2f}\n"
        "Naive diff-in-means on the flipped file should now be DECISIVELY NEGATIVE."
    )


if __name__ == "__main__":
    main()
