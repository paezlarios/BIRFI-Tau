"""Fisher and profile-deviance errors agree for one lifetime."""


def test_fisher_matches_profile_error(reference_result):
    """Acceptance: agree within 5 %."""
    r = reference_result
    assert abs(r.tau_fisher_error_ns / r.tau_profile_error_ns - 1) < 0.05
