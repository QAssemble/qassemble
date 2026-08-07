import math

from QAssemble.FLatDyn import G


def test_reference_nearest_selects_root_closest_to_previous_mu():
    def num_of_e(mu):
        return (mu + 1.0) * (mu - 0.1) * (mu - 2.0)

    sol, diagnostics = G._solve_reference_nearest(
        num_of_e,
        mu_reference=0.0,
        ecut=3.0,
        scan_points=121,
    )

    assert math.isclose(sol, 0.1, abs_tol=1.0e-6)
    assert diagnostics["local_root_found"] is True
    assert diagnostics["local_root_count"] == 3
    assert diagnostics["used_global_fallback"] is False


def test_reference_nearest_reports_missing_local_root():
    def num_of_e(mu):
        return mu - 5.0

    sol, diagnostics = G._solve_reference_nearest(
        num_of_e,
        mu_reference=0.0,
        ecut=1.0,
        scan_points=21,
    )

    assert sol is None
    assert diagnostics["local_root_found"] is False
    assert diagnostics["used_global_fallback"] is True
    assert diagnostics["local_root_count"] == 0


def test_reference_bisect_matches_gw_edmft_delta_mu_update_rule():
    def num_of_e(mu):
        return 1.0 - mu

    sol, diagnostics = G._solve_reference_bisect(
        num_of_e,
        mu_reference=0.0,
        ecut=4.0,
        density_tol=1.0e-10,
        max_iter=200,
    )

    assert math.isclose(sol, 1.0, abs_tol=1.0e-8)
    assert math.isclose(diagnostics["delta_mu"], 1.0, abs_tol=1.0e-8)
    assert diagnostics["search_mode"] == "reference_bisect"
    assert diagnostics["used_global_fallback"] is False
