"""Downloads the pretrained weights of the foundation models at the commits that the paper used and verifies them (sha256).

environment/weights.json lists the repositories, commits and files; docs/reproducing.md ("Foundation-model weights") covers access (TabPFN-3.5 is gated) and
licenses. Files already in the local Hugging Face cache at that commit are only verified; the others are downloaded by commit, never from `main`. Run it
before scripts/05_qualify_foundation.py and the benchmark, which stop if a pinned file is missing or differs.
Usage: python scripts/00_fetch_weights.py [--models TabPFN,TabICL,TabFM,TabDPT,EXAONE] [--verify-only]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # noqa: E402

from defectshift.models import weights  # noqa: E402

ALL = ("TabPFN", "TabICL", "TabFM", "TabDPT", "EXAONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(ALL), help="comma-separated models (default: all five)")
    ap.add_argument("--verify-only", action="store_true", help="only verify the files that are already in the cache; no download")
    a = ap.parse_args()
    names = a.models.split(",")
    unknown = sorted(set(names) - set(ALL))
    if unknown:
        raise SystemExit(f"unknown models {unknown}; choose from {list(ALL)}")
    bad = 0
    for n in names:
        try:
            if a.verify_only:
                snap = weights.snapshot_dir(n)
            else:
                snap = weights.fetch(n)
            print(f"[weights] {n:7s} ok  commit {weights.commit(n)[:8]}  {snap}")
        except Exception as e:                                   # gated repository, network, missing or different file
            bad += 1
            print(f"[weights] {n:7s} FAILED: {type(e).__name__}: {e}")
    if bad:
        raise SystemExit(f"{bad} of {len(names)} models are not available at the pinned commit")


if __name__ == "__main__":
    main()
