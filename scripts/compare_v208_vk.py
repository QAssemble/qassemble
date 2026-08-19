#!/usr/bin/env python3
"""Compare QAssemble's full bare interaction with gw_edmft_v208 V(k)."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from QAssemble.BLatStc import V
from QAssemble.Crystal import Crystal


def load_input(path: Path) -> dict:
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="QAssemble input.ini")
    parser.add_argument("--h5", type=Path, help="compare stored init/V/vk instead of rebuilt V(k)")
    parser.add_argument("--tol", type=float, default=1.0e-12)
    args = parser.parse_args()

    namespace = load_input(args.input)
    crystal = Crystal(cry=namespace["Crystal"])
    if crystal.ns != 1 or len(crystal.siteorbitals) != 1:
        raise ValueError("gw_edmft_v208 comparison expects the single-site, spin-summed model")

    twobody = namespace["Hamiltonian"]["TwoBody"]
    amplitudes = {
        float(amplitude)
        for pair, terms in twobody["NonLocal"].items()
        if isinstance(pair, tuple)
        for amplitude in terms
    }
    if len(amplitudes) != 1:
        raise ValueError(f"expected one nonlocal amplitude, got {sorted(amplitudes)}")
    v1 = amplitudes.pop()

    built = V(crystal=crystal, twobody=twobody)
    actual = built.k
    if args.h5 is not None:
        with h5py.File(args.h5, "r") as handle:
            actual = handle["init/V/vk"][...]
    if actual.shape != built.k.shape:
        raise ValueError(f"V(k) shape mismatch: {actual.shape} != {built.k.shape}")

    site = next(iter(crystal.siteorbitals))
    orbitals = crystal.siteorbitals[site]
    density = [crystal.BIndex([site, [orbital, orbital]]) for orbital in orbitals]

    expected = np.repeat(built.vloc.vloc[..., np.newaxis], crystal.nk, axis=4).astype(complex)
    form_factor = 2.0 * v1 * np.cos(2.0 * np.pi * crystal.kpoint).sum(axis=1)
    for iorb in density:
        for jorb in density:
            expected[iorb, jorb, 0, 0, :] += form_factor

    delta = actual - expected
    abs_delta = np.abs(delta)
    worst = np.unravel_index(np.argmax(abs_delta), abs_delta.shape)
    iorb, jorb, _, _, ik = worst
    hermitian_error = np.max(np.abs(actual - np.conjugate(actual.swapaxes(0, 1))))
    kzero = int(np.flatnonzero(np.all(np.isclose(crystal.kpoint, 0.0), axis=1))[0])
    density_ix = np.ix_(density, density, [0], [0], [kzero])

    np.set_printoptions(precision=12, suppress=True)
    print(f"grid={tuple(crystal.rkgrid)} nk={crystal.nk} product_basis={actual.shape[0]} V1={v1}")
    print("QAssemble density block at k=0:")
    print(actual[density_ix][:, :, 0, 0, 0].real)
    print("gw_edmft_v208 density block at k=0:")
    print(expected[density_ix][:, :, 0, 0, 0].real)
    print(f"max_abs_error={abs_delta[worst]:.12e}")
    print(
        "worst="
        f"k_index={ik} k={crystal.kpoint[ik].tolist()} "
        f"component=({iorb},{jorb}) actual={actual[worst]} expected={expected[worst]}"
    )
    print(f"hermitian_error={hermitian_error:.12e}")

    if abs_delta[worst] > args.tol:
        print(f"FAIL: full V(k) differs from gw_edmft_v208 (tol={args.tol:.1e})")
        return 1
    print(f"PASS: all {crystal.nk} k-points and {actual.shape[0]}x{actual.shape[1]} components match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
