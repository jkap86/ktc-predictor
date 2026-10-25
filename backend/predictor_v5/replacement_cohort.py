"""Cross-source statistics consistency checks."""

from predictor_v5.replacement_mfl import STAT_COLUMNS


def verify_common_stats(ktc, mfl):
    keys = ["player_id", "season"]
    for frame in (ktc, mfl):
        if (
            not set(STAT_COLUMNS).issubset(frame.columns)
            or frame[keys].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError("invalid stats keys")
    joined = ktc[STAT_COLUMNS].merge(
        mfl[STAT_COLUMNS], on=keys, suffixes=("_ktc", "_mfl"), validate="one_to_one"
    )
    if any(
        not joined[c + "_ktc"].eq(joined[c + "_mfl"]).all() for c in STAT_COLUMNS[2:]
    ):
        raise ValueError("common stats disagree")
