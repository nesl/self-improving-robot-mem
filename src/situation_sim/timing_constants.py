"""Numeric timing constants, COPIED from
``src/households/realization_params.yaml`` (values as of commit 8cc19a14c).
Copied rather than imported so this package never loads the old simulator.
Every value below keeps the old file's calibration note in one line.
"""

# Day-to-day sigma (minutes) of block starts, by jitter class.
# Source: casas/README.md "Jitter calibration" (CASAS aruba/cairo/milan/
# tulum2 anchor-start sigmas; `external` is a domain prior).
JITTER_CLASSES_MIN = {
    "external": 10,
    "routine": 30,
    "flexible": 75,
    "loose": 110,
}

# Per-resident punctuality multiplier bounds (revamp_v1 README:
# "hard-bounded to [0.5, 2.0]").
JITTER_SCALE_MIN = 0.5
JITTER_SCALE_MAX = 2.0

# A realized block keeps at least this share of its authored length; a
# shifted block pushes the next one instead of collapsing it.
KEEP_BLOCK_SHARE = 0.6

# Bout fragmentation: N ~ max(1, Poisson(mean_bouts)) sub-bouts within the
# realized block window; bounds at session scale (CASAS session-merge).
MIN_MEAN_BOUTS = 1.0
MAX_MEAN_BOUTS = 6.0
# A sub-bout shorter than this collapses into its neighbour.
MIN_BOUT_MINUTES = 10

# Upper bound on a per-realization block drop probability.
MAX_SKIP_P = 0.9

# Standing (item, trip-type) carry omissions: with 0.85 roughly one pairing
# in seven is a standing omission. Never for phone/keys; wallet only on short
# local trips (see Simulator.standing_omission).
CARRY_P = 0.85
# Per-departure forgetting of a pocket item, by the owner's forgetfulness.
FORGET_LEVELS = {"rarely": 0.01, "sometimes": 0.03, "often": 0.08}
FORGET_P_DEFAULT = 0.03
