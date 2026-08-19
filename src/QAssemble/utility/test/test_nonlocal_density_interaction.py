import numpy as np
import pytest

from QAssemble.BLatStc import V
from QAssemble.Crystal import Crystal


V_NN = 0.2
POSITIVE_NEIGHBORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
FULL_NEIGHBORS = POSITIVE_NEIGHBORS + [
    [-value for value in neighbor] for neighbor in POSITIVE_NEIGHBORS
]


def _three_orbital_crystal(kgrid=(4, 4, 4)):
    return Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 3]],
            "NSpin": 1,
            "NElec": 3.0,
            "KGrid": list(kgrid),
        }
    )


def _twobody(nonlocal_terms):
    return {
        "Local": {
            "Parameter": "SlaterKanamori",
            "option": {
                (0, (0, 1, 2)): {"U": 1.0, "Up": 0.8, "J": 0.0, "l": 1},
            },
        },
        "NonLocal": nonlocal_terms,
    }


def _all_density_pairs(neighbors):
    return {
        ((0, iorb), (0, jorb)): {V_NN: neighbors}
        for iorb in range(3)
        for jorb in range(iorb, 3)
    }


def _density_indices(crystal):
    return [crystal.BIndex([0, [iorb, iorb]]) for iorb in range(3)]


def test_directed_input_adds_only_hermitian_reverse_bond():
    crystal = _three_orbital_crystal(kgrid=(4, 1, 1))
    amplitude = V_NN + 0.05j
    terms = {
        ((0, 0), (0, 1)): {amplitude: [[1, 0, 0]]},
    }
    built = V(crystal=crystal, twobody=_twobody(terms))
    density = _density_indices(crystal)
    plus_x, minus_x = 1, 3

    expected_forward = np.zeros(4, dtype=np.complex128)
    expected_forward[plus_x] = amplitude
    expected_reverse = np.zeros(4, dtype=np.complex128)
    expected_reverse[minus_x] = np.conjugate(amplitude)
    np.testing.assert_allclose(
        built.nonlocr[density[0], density[1], 0, 0], expected_forward
    )
    np.testing.assert_allclose(
        built.nonlocr[density[1], density[0], 0, 0], expected_reverse
    )


def test_distinct_plus_minus_r_amplitudes_are_allowed_and_hermitian():
    crystal = _three_orbital_crystal(kgrid=(4, 1, 1))
    plus_amplitude = 0.2
    minus_amplitude = 0.3
    terms = {
        ((0, 0), (0, 1)): {
            plus_amplitude: [[1, 0, 0]],
            minus_amplitude: [[-1, 0, 0]],
        },
    }
    built = V(crystal=crystal, twobody=_twobody(terms))
    density = _density_indices(crystal)
    plus_x, minus_x = 1, 3

    np.testing.assert_allclose(
        built.nonlocr[density[0], density[1], 0, 0, [plus_x, minus_x]],
        [plus_amplitude, minus_amplitude],
    )
    np.testing.assert_allclose(
        built.nonlocr[density[1], density[0], 0, 0, [plus_x, minus_x]],
        [minus_amplitude, plus_amplitude],
    )
    np.testing.assert_allclose(
        built.k, np.conjugate(built.k.swapaxes(0, 1)), atol=1.0e-12, rtol=0.0
    )


def test_three_band_nonlocal_interaction_matches_cosine_on_full_product_basis():
    crystal = _three_orbital_crystal()
    built = V(
        crystal=crystal,
        twobody=_twobody(_all_density_pairs(FULL_NEIGHBORS)),
    )
    density = _density_indices(crystal)
    form_factor = 2.0 * V_NN * np.cos(2.0 * np.pi * crystal.kpoint).sum(axis=1)
    expected = np.repeat(
        built.vloc.vloc[..., np.newaxis], crystal.nk, axis=4
    ).astype(np.complex128)
    for iorb in density:
        for jorb in density:
            expected[iorb, jorb, 0, 0, :] += form_factor

    np.testing.assert_allclose(built.k, expected, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        built.k, np.conjugate(built.k.swapaxes(0, 1)), atol=1.0e-12, rtol=0.0
    )


def test_three_band_gamma_density_block_has_intended_u_up_and_v():
    crystal = _three_orbital_crystal()
    built = V(
        crystal=crystal,
        twobody=_twobody(_all_density_pairs(FULL_NEIGHBORS)),
    )
    density = _density_indices(crystal)

    # This regression targets non-local assembly, so impose the model's local
    # density block directly instead of retesting the independent VLoc builder.
    for row, iorb in enumerate(density):
        for col, jorb in enumerate(density):
            built.vloc.vloc[iorb, jorb, 0, 0] = 1.0 if row == col else 0.8
    built.LocPlusNonLoc()

    gamma = int(np.flatnonzero(np.all(np.isclose(crystal.kpoint, 0.0), axis=1))[0])
    block = built.k[np.ix_(density, density, [0], [0], [gamma])][:, :, 0, 0, 0]
    expected = np.full((3, 3), 2.0)
    np.fill_diagonal(expected, 2.2)
    np.testing.assert_allclose(block, expected, atol=1.0e-12, rtol=0.0)


def test_explicit_hermitian_reverse_is_idempotent_and_conflicts_are_rejected():
    crystal = _three_orbital_crystal(kgrid=(4, 1, 1))
    amplitude = V_NN + 0.05j
    reference = V(
        crystal=crystal,
        twobody=_twobody(
            {((0, 0), (0, 1)): {amplitude: [[1, 0, 0]]}}
        ),
    )
    explicit = V(
        crystal=crystal,
        twobody=_twobody(
            {
                ((0, 0), (0, 1)): {amplitude: [[1, 0, 0]]},
                ((0, 1), (0, 0)): {
                    np.conjugate(amplitude): [[-1, 0, 0]]
                },
            }
        ),
    )

    np.testing.assert_allclose(
        explicit.nonlocr, reference.nonlocr, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(explicit.k, reference.k, atol=0.0, rtol=0.0)

    conflicting = {
        ((0, 0), (0, 1)): {
            amplitude: [[1, 0, 0]],
        },
        ((0, 1), (0, 0)): {
            0.3: [[-1, 0, 0]],
        },
    }

    with pytest.raises(ValueError, match="Conflicting non-local density interaction"):
        V(crystal=crystal, twobody=_twobody(conflicting))
