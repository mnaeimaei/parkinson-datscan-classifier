from __future__ import annotations

import math


HIGH_CONFIDENCE_DICE = 0.70
FAILED_DICE = 0.50

SMALL_GAIN_THRESHOLD = 0.05
INITIAL_GAIN_THRESHOLD = 0.10


def qc_category(dice: float) -> str:
    if dice >= HIGH_CONFIDENCE_DICE:
        return "high_confidence"

    if dice >= FAILED_DICE:
        return "review"

    return "failed"


def _validate_dice(name: str, value: float) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} is not finite: {value}"
        )

    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"{name} outside [0, 1]: {value}"
        )

    return value


def select_registration_candidate(
    *,
    moments_initial_dice: float,
    geometry_initial_dice: float,
    moments_optimized_dice: float,
    geometry_optimized_dice: float,
) -> dict:
    """
    Deterministic inference-time registration policy.

    Uses only the four candidate Dice values.

    No:
      - UID-specific decisions
      - manual QC
      - saved Step-6B4 manifest
      - labels
    """

    moments_initial_dice = _validate_dice(
        "moments_initial_dice",
        moments_initial_dice,
    )

    geometry_initial_dice = _validate_dice(
        "geometry_initial_dice",
        geometry_initial_dice,
    )

    moments_optimized_dice = _validate_dice(
        "moments_optimized_dice",
        moments_optimized_dice,
    )

    geometry_optimized_dice = _validate_dice(
        "geometry_optimized_dice",
        geometry_optimized_dice,
    )

    # ---------------------------------------------------------
    # Best initial
    #
    # Preserve Step-6B4 tie behavior:
    # moments wins when Dice values are equal.
    # ---------------------------------------------------------

    if moments_initial_dice >= geometry_initial_dice:
        best_initial_candidate = "moments_initial"
        best_initial_dice = moments_initial_dice
    else:
        best_initial_candidate = "geometry_initial"
        best_initial_dice = geometry_initial_dice

    # ---------------------------------------------------------
    # Best optimized
    # ---------------------------------------------------------

    if moments_optimized_dice >= geometry_optimized_dice:
        best_optimized_candidate = "moments_optimized"
        best_optimized_dice = moments_optimized_dice
    else:
        best_optimized_candidate = "geometry_optimized"
        best_optimized_dice = geometry_optimized_dice

    gain = (
        best_initial_dice
        - best_optimized_dice
    )

    # =========================================================
    # CASE 1
    # Catastrophic optimizer failure
    # =========================================================

    if (
        best_optimized_dice < FAILED_DICE
        and best_initial_dice >= HIGH_CONFIDENCE_DICE
    ):
        return {
            "status": "selected",
            "reason": "catastrophic_optimizer_failure",

            "selected_candidate":
                best_initial_candidate,

            "selected_dice":
                best_initial_dice,

            "selected_qc":
                qc_category(best_initial_dice),

            "best_initial_candidate":
                best_initial_candidate,

            "best_initial_dice":
                best_initial_dice,

            "best_optimized_candidate":
                best_optimized_candidate,

            "best_optimized_dice":
                best_optimized_dice,

            "initial_gain_over_optimized":
                gain,

            "requires_similarity_rescue":
                False,
        }

    # =========================================================
    # CASE 2
    # All rigid candidates failed
    # =========================================================

    if (
        max(
            best_initial_dice,
            best_optimized_dice,
        )
        < FAILED_DICE
    ):
        best_failed_dice = max(
            best_initial_dice,
            best_optimized_dice,
        )

        return {
            "status":
                "similarity_rescue_required",

            "reason":
                "all_rigid_candidates_failed",

            "selected_candidate":
                None,

            "selected_dice":
                best_failed_dice,

            "selected_qc":
                "failed",

            "best_initial_candidate":
                best_initial_candidate,

            "best_initial_dice":
                best_initial_dice,

            "best_optimized_candidate":
                best_optimized_candidate,

            "best_optimized_dice":
                best_optimized_dice,

            "initial_gain_over_optimized":
                gain,

            "requires_similarity_rescue":
                True,
        }

    # =========================================================
    # CASE 3
    # Initial transform substantially better
    #
    # Offline Step-6B4:
    #     manual_review
    #
    # Step-7I inference adaptation:
    #     automatically use best initial.
    #
    # Validated by Step 7I-B1:
    #     25 / 25 manual cases reproduced exactly.
    # =========================================================

    if gain >= INITIAL_GAIN_THRESHOLD:
        return {
            "status": "selected",

            "reason":
                "large_initial_gain_use_initial",

            "selected_candidate":
                best_initial_candidate,

            "selected_dice":
                best_initial_dice,

            "selected_qc":
                qc_category(best_initial_dice),

            "best_initial_candidate":
                best_initial_candidate,

            "best_initial_dice":
                best_initial_dice,

            "best_optimized_candidate":
                best_optimized_candidate,

            "best_optimized_dice":
                best_optimized_dice,

            "initial_gain_over_optimized":
                gain,

            "requires_similarity_rescue":
                False,
        }

    # =========================================================
    # CASE 4
    # Moderate initial advantage
    #
    # Preserve original Step-6B4 policy:
    # keep optimized transform.
    # =========================================================

    if gain >= SMALL_GAIN_THRESHOLD:
        return {
            "status": "selected",

            "reason":
                "moderate_initial_gain_keep_optimized",

            "selected_candidate":
                best_optimized_candidate,

            "selected_dice":
                best_optimized_dice,

            "selected_qc":
                qc_category(best_optimized_dice),

            "best_initial_candidate":
                best_initial_candidate,

            "best_initial_dice":
                best_initial_dice,

            "best_optimized_candidate":
                best_optimized_candidate,

            "best_optimized_dice":
                best_optimized_dice,

            "initial_gain_over_optimized":
                gain,

            "requires_similarity_rescue":
                False,
        }

    # =========================================================
    # CASE 5
    # Small / negligible initial advantage, or optimized better
    # =========================================================

    return {
        "status": "selected",

        "reason":
            "small_gain_keep_optimized",

        "selected_candidate":
            best_optimized_candidate,

        "selected_dice":
            best_optimized_dice,

        "selected_qc":
            qc_category(best_optimized_dice),

        "best_initial_candidate":
            best_initial_candidate,

        "best_initial_dice":
            best_initial_dice,

        "best_optimized_candidate":
            best_optimized_candidate,

        "best_optimized_dice":
            best_optimized_dice,

        "initial_gain_over_optimized":
            gain,

        "requires_similarity_rescue":
            False,
    }
