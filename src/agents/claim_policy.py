from __future__ import annotations

from dataclasses import dataclass

MAX_CLAIMS_PER_RUN = 20
MAX_CLAIMS_PER_BRANCH = 5
MAX_SUBCLAIMS_PER_CLAIM = 2
MAX_SUBCLAIM_DEPTH = 1


@dataclass(frozen=True)
class ClaimLimitSettings:
    max_claims_per_run: int = MAX_CLAIMS_PER_RUN
    max_claims_per_branch: int = MAX_CLAIMS_PER_BRANCH
    max_subclaims_per_claim: int = MAX_SUBCLAIMS_PER_CLAIM
    max_subclaim_depth: int = MAX_SUBCLAIM_DEPTH


def claim_limit_settings_from_object(settings: object | None = None) -> ClaimLimitSettings:
    return ClaimLimitSettings(
        max_claims_per_run=_int_setting(
            settings,
            "max_claims_per_run",
            "MAX_CLAIMS_PER_RUN",
            MAX_CLAIMS_PER_RUN,
        ),
        max_claims_per_branch=_int_setting(
            settings,
            "max_claims_per_branch",
            "MAX_CLAIMS_PER_BRANCH",
            MAX_CLAIMS_PER_BRANCH,
        ),
        max_subclaims_per_claim=_int_setting(
            settings,
            "max_subclaims_per_claim",
            "MAX_SUBCLAIMS_PER_CLAIM",
            MAX_SUBCLAIMS_PER_CLAIM,
        ),
        max_subclaim_depth=_int_setting(
            settings,
            "max_subclaim_depth",
            "MAX_SUBCLAIM_DEPTH",
            MAX_SUBCLAIM_DEPTH,
        ),
    )


def _int_setting(
    settings: object | None,
    lower_name: str,
    upper_name: str,
    default: int,
) -> int:
    if settings is None:
        return default
    for name in (lower_name, upper_name):
        if not hasattr(settings, name):
            continue
        try:
            return max(0, int(getattr(settings, name)))
        except (TypeError, ValueError):
            return default
    return default
