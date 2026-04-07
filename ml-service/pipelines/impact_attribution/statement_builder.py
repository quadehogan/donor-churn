import math

OUTCOME_LABELS = {
    'education': 'average education progress',
    'health':    'average health and wellbeing scores',
}

DIRECTION_WORDS = {
    True:  'improvement in',
    False: 'decline in',
}


def build_impact_statement(
    supporter_id: str,
    safehouse_name: str,
    program_area: str,
    amount: float,
    outcome: str,
    coef: float,
    ci_low: float,
    ci_high: float,
    window: int,
    baseline: float,
) -> str:
    """
    Generates a single personalized impact statement for a donor.
    Only called when the underlying effect is statistically significant.
    """
    log_amount     = math.log1p(amount)
    estimated_pct  = coef * log_amount
    direction      = estimated_pct >= 0
    direction_word = DIRECTION_WORDS[direction]
    outcome_label  = OUTCOME_LABELS.get(outcome, outcome)

    ci_low_abs  = abs(ci_low  * log_amount)
    ci_high_abs = abs(ci_high * log_amount)
    # Ensure ci range is ordered low → high in the statement
    range_lo = min(ci_low_abs, ci_high_abs)
    range_hi = max(ci_low_abs, ci_high_abs)

    return (
        f"Your ${amount:,.0f} allocated to {program_area} at {safehouse_name} "
        f"was associated with a {abs(estimated_pct):.1f}% {direction_word} "
        f"{outcome_label} over the following {window} months "
        f"(estimated range: {range_lo:.1f}%\u2013{range_hi:.1f}%)."
    )
