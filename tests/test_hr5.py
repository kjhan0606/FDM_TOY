from __future__ import annotations

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from fdm_smbh_delay.hr5 import (
    MKAGN_DTYPE,
    binned_source_rate,
    bootstrap_binned_source_rate,
    bootstrap_redshift_rate,
    circular_gw_background_contributions,
    cumulative_active_sources,
    delayed_redshift,
    fibonacci_sightlines,
    fit_redshift_rate,
    find_agn_pair_population,
    find_dual_agn_pairs,
    histogram_quantiles,
    infer_capture_receivers,
    interval_censored_cumulative_bounds,
    pair_component_multiplicity,
    project_pair_observables,
    read_mkagn_snapshot,
    redshift_rate_model,
)


@pytest.fixture
def cosmology() -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=68.4, Om0=0.3, Tcmb0=2.7255)


def test_redshift_rate_fit_recovers_synthetic_curve() -> None:
    redshift = np.geomspace(0.2, 8.0, 30)
    expected = (2.0e-11, 2.3, 1.0, 4.2)
    rate = redshift_rate_model(redshift, *expected)
    fit = fit_redshift_rate(redshift, rate, np.full(redshift.size, 100))
    assert fit.success
    assert fit.phi_star == pytest.approx(expected[0], rel=2.0e-4)
    assert fit.z_star == pytest.approx(expected[1], rel=2.0e-4)
    assert fit.alpha == pytest.approx(expected[2], rel=2.0e-4)
    assert fit.beta == pytest.approx(expected[3], rel=2.0e-4)


def test_redshift_rate_bootstrap_is_reproducible_and_ordered() -> None:
    redshift = np.geomspace(0.2, 8.0, 24)
    expected = (2.0e-7, 2.3, 1.0, 4.2)
    exposure = np.full(redshift.size, 1.0e9)
    count = np.rint(redshift_rate_model(redshift, *expected) * exposure).astype(int)
    first = bootstrap_redshift_rate(
        redshift, count, exposure, 40, np.random.default_rng(1729)
    )
    second = bootstrap_redshift_rate(
        redshift, count, exposure, 40, np.random.default_rng(1729)
    )
    assert first.shape == (40, 4)
    assert np.array_equal(first, second)
    quantiles = np.quantile(first, (0.16, 0.5, 0.84), axis=0)
    assert np.all(quantiles[0] <= quantiles[1])
    assert np.all(quantiles[1] <= quantiles[2])
    assert quantiles[1, 0] == pytest.approx(expected[0], rel=0.08)


def test_binned_source_rate_bootstrap_is_reproducible_and_ordered() -> None:
    count = np.array([0, 4, 100])
    exposure = np.array([2.0, 2.0, 5.0])
    first = bootstrap_binned_source_rate(
        count, exposure, 400, np.random.default_rng(314159)
    )
    second = bootstrap_binned_source_rate(
        count, exposure, 400, np.random.default_rng(314159)
    )
    assert first.shape == (3, 3)
    assert np.array_equal(first, second)
    assert np.all(first[0] <= first[1])
    assert np.all(first[1] <= first[2])
    assert np.all(first[:, 0] == 0.0)
    assert first[1, 2] == pytest.approx(count[2] / exposure[2], rel=0.05)


def test_fixed_delay_moves_events_to_lower_redshift(cosmology: FlatLambdaCDM) -> None:
    capture_redshift = np.array([1.0, 3.0])
    capture_time = np.asarray(cosmology.age(capture_redshift).value)
    shifted, censored = delayed_redshift(capture_time, 0.5, cosmology, grid_size=10000)
    assert np.all(shifted < capture_redshift)
    assert not np.any(censored)


def test_source_rate_and_active_count_are_positive(cosmology: FlatLambdaCDM) -> None:
    edges = np.array([0.5, 1.0, 2.0])
    events = np.array([0.6, 0.8, 1.2, 1.5, 1.8])
    count, rate, error = binned_source_rate(events, edges, 1.0e6, cosmology)
    assert count.tolist() == [2, 3]
    assert np.all(rate > 0.0)
    assert np.all(error > 0.0)

    z = np.linspace(0.0, 2.0, 100)
    cumulative = cumulative_active_sources(z, np.full_like(z, 1.0e-12), 1.0e4, cosmology)
    assert cumulative[0] == 0.0
    assert np.all(np.diff(cumulative) >= 0.0)


def test_circular_gw_background_contribution_scaling() -> None:
    mass = np.array([1.0e8, 2.0e8])
    redshift = np.array([1.0, 1.0])
    reference_frequency = 1.0 / (365.25 * 86400.0)
    contribution = circular_gw_background_contributions(
        mass,
        redshift,
        volume_cmpc3=1.0e6,
        observed_frequency_hz=reference_frequency,
    )
    assert np.all(contribution > 0.0)
    assert contribution[1] / contribution[0] == pytest.approx(2.0 ** (5.0 / 3.0))

    lower_frequency = circular_gw_background_contributions(
        mass[:1],
        redshift[:1],
        volume_cmpc3=1.0e6,
        observed_frequency_hz=reference_frequency / 2.0,
    )
    assert lower_frequency[0] / contribution[0] == pytest.approx(2.0 ** (4.0 / 3.0))


def test_circular_gw_background_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        circular_gw_background_contributions(
            np.array([0.0]), np.array([1.0]), 1.0e6, 1.0e-8
        )


def test_histogram_quantiles() -> None:
    count = np.array([[0, 1, 8, 1, 0], [0, 0, 0, 0, 0]])
    result = histogram_quantiles(count, np.arange(6.0), (0.16, 0.5, 0.84))
    assert result[0, 1] == pytest.approx(2.0)
    assert np.all(np.isnan(result[1]))


def test_read_mkagn_snapshot(tmp_path) -> None:
    path = tmp_path / "agn.00020.dat"
    records = np.zeros(2, dtype=MKAGN_DTYPE)
    records["sink_id"] = [3, 8]
    records["mass"] = [1.0e4, 2.0e4]
    with path.open("wb") as stream:
        np.array([9.5], dtype="<f8").tofile(stream)
        np.array([2.0e4], dtype="<f8").tofile(stream)
        np.array([2], dtype="<i4").tofile(stream)
        records.tofile(stream)
    redshift, timestep, loaded = read_mkagn_snapshot(path)
    assert MKAGN_DTYPE.itemsize == 360
    assert redshift == pytest.approx(9.5)
    assert timestep == pytest.approx(2.0e4)
    assert loaded["sink_id"].tolist() == [3, 8]
    assert loaded["mass"].tolist() == [1.0e4, 2.0e4]


def test_infer_capture_receiver_uses_mass_cut_and_periodic_distance() -> None:
    receiver = infer_capture_receivers(
        minor_id=np.array([1, 2]),
        minor_mass=np.array([10.0, 20.0]),
        minor_position=np.array([[0.01, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        current_id=np.array([10, 11, 12]),
        current_mass=np.array([15.0, 25.0, 50.0]),
        current_position=np.array([[9.99, 0.0, 0.0], [0.02, 0.0, 0.0], [4.01, 0.0, 0.0]]),
        box_size_cmpc_over_h=10.0,
        maximum_radius_cmpc_over_h=0.1,
    )
    assert receiver.tolist() == [11, 12]


def test_find_dual_agn_pairs_applies_activity_and_physical_separation() -> None:
    records = np.zeros(4, dtype=MKAGN_DTYPE)
    records["sink_id"] = [1, 2, 3, 4]
    records["mass"] = [6.84e6, 3.42e6, 6.84e6, 6.84e6]
    records["Lbol"] = [2.0e43, 3.0e43, 1.0e42, 4.0e43]
    records["x"] = [0.01, 0.02, 0.03, 9.99]
    pairs = find_dual_agn_pairs(
        records,
        redshift=0.0,
        dimensionless_hubble=1.0,
        luminosity_threshold_erg_s=1.0e43,
        minimum_separation_pkpc=5.0,
        maximum_separation_pkpc=25.0,
        box_size_cmpc_over_h=10.0,
    )
    assert int(pairs["active_count"]) == 3
    assert sorted(zip(pairs["id_1"], pairs["id_2"])) == [(1, 2), (1, 4)]
    assert np.allclose(np.sort(pairs["separation_pkpc"]), [10.0, 20.0])


def test_find_agn_pair_population_separates_dual_and_offset_pairs() -> None:
    records = np.zeros(4, dtype=MKAGN_DTYPE)
    records["sink_id"] = [1, 2, 3, 4]
    records["mass"] = [4.0e7, 2.0e7, 1.0e7, 1.0e7]
    records["Lbol"] = [2.0e43, 3.0e43, 1.0e42, 1.0e42]
    records["LhX"] = records["Lbol"] / 20.0
    records["x"] = [0.010, 0.020, 0.030, 1.000]
    pairs = find_agn_pair_population(
        records,
        redshift=0.0,
        dimensionless_hubble=1.0,
        maximum_separation_pkpc=25.0,
        box_size_cmpc_over_h=10.0,
    )
    assert int(pairs["active_count"]) == 2
    assert pairs["id_1"].tolist() == [1, 1, 2]
    assert pairs["id_2"].tolist() == [2, 3, 3]
    assert pairs["is_dual"].tolist() == [True, False, False]
    assert pairs["is_offset"].tolist() == [False, True, True]
    assert np.all(pairs["mass_1_msun"] >= pairs["mass_2_msun"])


def test_pair_component_multiplicity_identifies_multiple_system() -> None:
    pair_size, member, member_size = pair_component_multiplicity(
        np.array([1, 2, 8]), np.array([2, 3, 9])
    )
    assert pair_size.tolist() == [3, 3, 2]
    assert dict(zip(member.tolist(), member_size.tolist())) == {
        1: 3,
        2: 3,
        3: 3,
        8: 2,
        9: 2,
    }


def test_project_pair_observables_includes_hubble_flow() -> None:
    position_1 = np.zeros((1, 3))
    position_2 = np.array([[0.003, 0.004, 0.0]])
    velocity_1 = np.zeros((1, 3))
    velocity_2 = np.array([[10.0, 20.0, 0.0]])
    projected, line_velocity = project_pair_observables(
        position_1,
        position_2,
        velocity_1,
        velocity_2,
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        redshift=0.0,
        dimensionless_hubble=1.0,
        hubble_kms_mpc=100.0,
        box_size_cmpc_over_h=10.0,
    )
    assert projected[0].tolist() == pytest.approx([4.0, 5.0])
    assert line_velocity[0].tolist() == pytest.approx([10.3, 0.0])
    sightlines = fibonacci_sightlines(64)
    assert np.allclose(np.linalg.norm(sightlines, axis=1), 1.0)


def test_interval_censored_cumulative_bounds_preserve_censoring() -> None:
    lower, upper = interval_censored_cumulative_bounds(
        np.array([0.0, 0.8, np.nan]),
        np.array([0.2, 1.0, np.nan]),
        np.array([0.0, 0.5, 0.9, 1.0, 2.0]),
        followup_gyr=1.0,
    )
    assert lower[:4].tolist() == pytest.approx([0.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0])
    assert upper[:4].tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert np.isnan(lower[-1])
    assert np.isnan(upper[-1])
