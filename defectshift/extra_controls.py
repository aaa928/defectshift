"""Seeds and target fold counts of the additional randomized lot-split controls.
The original ten controls (seeds 9001-9010) get their target fold count m = round(100 |B| / N) from the |B| targets of
config.PLACEBO_B_TARGET. Changing the shared configuration file (config.py) would break the compatibility checks of existing runs, so the
additional seeds are registered in the same table at run time (register()). Their |B| target is derived backwards from the fold count m:
T = round(m x N / 100), so that round(100 T / N) = m; the split generation (candidates.build_one) is the same as for the original controls.

Groups: small = the target share of cat_050 (23.7%, m = 24); each large group = the target share of one of the other domains. Seeds are
assigned consecutively within a group.
"""
from defectshift import config

N_ROWS = 14276

# (group name, target fold count m, seed range [start, end] inclusive)
PLAN = (
    ("small_cat_050", 24, (9011, 9030)),
    ("large_cat_045", 43, (9031, 9034)),
    ("large_cat_005", 49, (9035, 9038)),
    ("large_x_time_y_time", 50, (9039, 9042)),
    ("large_cat_019", 55, (9043, 9046)),
    ("large_cat_039", 58, (9047, 9050)),
)


def seed_table(plan=PLAN):
    """{seed: (group, m)}. Stops if a seed repeats or collides with an original control seed (9001-9010)."""
    out = {}
    for group, m, (a, b) in plan:
        for s in range(a, b + 1):
            if s in out or s in range(9001, 9011):
                raise SystemExit(f"control seed {s} is duplicated")
            out[s] = (group, m)
    return out


def register(plan=PLAN):
    """Register the additional seeds in config.PLACEBO_B_TARGET (checks that an existing entry has the same value)."""
    for s, (_, m) in seed_table(plan).items():
        t = int(round(m * N_ROWS / config.PLACEBO_N_FOLDS))
        if config.PLACEBO_B_TARGET.get(s, t) != t:
            raise SystemExit(f"control seed {s} is already registered with a different target")
        config.PLACEBO_B_TARGET[s] = t
