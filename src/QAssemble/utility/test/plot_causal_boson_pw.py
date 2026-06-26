"""Plot P/W bosonic causal projection against causal_boson.py.

This script is a graph-based check, not a pytest test.  It follows the P and W
workflow from ``TestCausalFermion.ipynb`` and overlays:

* the original DLR data,
* the noisy uniform-grid input,
* QAssemble's ``BLatDyn.CausalProjection`` output,
* the directly imported ``causal_boson.py`` reference output.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from QAssemble.BLatDyn import P, W
from QAssemble.BLatStc import V
from QAssemble.Crystal import Crystal
from QAssemble.FLatDyn import G0
from QAssemble.utility.DLR import DLR


REFERENCE_CAUSAL_BOSON = Path("/Users/moseongjun/usr/FullGWEDMFT/bin/causal_boson.py")


def single_band_hubbard() -> tuple[Crystal, DLR, np.ndarray]:
    crystal = Crystal(
        {
            "RVec": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "Basis": [[[0, 0, 0], 1]],
            "NSpin": 1,
            "NElec": 1.0,
            "KGrid": [2, 2, 2],
        }
    )
    kx, ky, kz = crystal.kpoint[:, 0], crystal.kpoint[:, 1], crystal.kpoint[:, 2]
    eps = -2.0 * (
        np.cos(2.0 * np.pi * kx)
        + np.cos(2.0 * np.pi * ky)
        + np.cos(2.0 * np.pi * kz)
    )
    hamtb = np.zeros((1, 1, 1, crystal.nk), dtype=np.complex128, order="F")
    hamtb[0, 0, 0, :] = eps
    dlr = DLR({"beta": 20.0, "cutoff": 50.0, "eps": 1.0e-12})
    return crystal, dlr, hamtb


def single_band_vbare(crystal: Crystal) -> V:
    twobody = {
        "Local": {
            "Parameter": "SlaterKanamori",
            "option": {
                (0, 0): {"U": 1.0, "J": 0.0, "l": 0},
            },
        },
        "NonLocal": {
            ((0, 0), (0, 0)): {
                0.2: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        },
    }
    return V(crystal=crystal, twobody=twobody, hdf5file=None)


def load_reference_causal_boson(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("plot_reference_causal_boson", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boson_lat_dlr_to_uniform(dlr: DLR, bf6: np.ndarray) -> np.ndarray:
    nu_uniform = dlr.MatsubaraBosonUniform()
    out = np.empty(bf6.shape[:-1] + (len(nu_uniform),), dtype=np.complex128, order="F")
    for ik in range(bf6.shape[4]):
        out[:, :, :, :, ik, :] = dlr.MatsubaraDLR2UniformGrid(
            bf6[:, :, :, :, ik, :],
            sign=1,
        )
    return np.asfortranarray(out)


def add_low_frequency_noise(
    data: np.ndarray,
    nu: np.ndarray,
    *,
    cutoff: float,
    norm: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(data, dtype=np.complex128, copy=True, order="F")
    indices = np.flatnonzero(np.abs(nu) < cutoff * 2.0 / 3.0)
    for idx in indices:
        out[..., idx] += norm * (
            rng.uniform(low=-1.0, high=1.0, size=out.shape[:-1])
            + 1j * rng.uniform(low=-1.0, high=1.0, size=out.shape[:-1])
        )
    return np.asfortranarray(out)


def reference_causal_projection(
    owner: P | W,
    uniform_data: np.ndarray,
    causal_boson,
    *,
    coefficient_sign: int,
    reflection_symmetry: bool,
    constraint_tol: float,
) -> np.ndarray:
    """Mirror ``BLatDyn.CausalProjection`` using imported causal_boson.py."""

    dlr = owner.dlr
    nu_uniform = np.asarray(dlr.MatsubaraBosonUniform(), dtype=np.float64)
    moments, high = owner.Moment(uniform_data, grid="uniform")

    norb, _, ns, _, nk, _ = uniform_data.shape
    converted = np.zeros((norb, norb, ns, ns, nk, len(dlr.nu)), dtype=np.complex128)
    for ik in range(nk):
        converted[:, :, :, :, ik, :] = dlr.MatsubaraUniformGrid2DLR(
            uniform_data[:, :, :, :, ik, :],
            omega=nu_uniform,
            sign=1,
        )

    ref_projector = causal_boson.BosonPoleQPProjector(
        beta=dlr.beta,
        fit_omega=dlr.nu,
        output_omega=dlr.nu,
        eps=dlr.eps,
        dlr_lambda=float(dlr.dB.lamb),
        reflection_symmetry=reflection_symmetry,
        coefficient_sign=coefficient_sign,
        constraint_tol=constraint_tol,
    )
    out = np.array(converted, dtype=np.complex128, copy=True, order="F")
    for ik in range(nk):
        for ispin in range(ns):
            for iorb in range(norb):
                c0 = float(np.real(high[iorb, iorb, ispin, ispin, ik]))
                c1, c2, _ = np.real(moments[iorb, iorb, ispin, ispin, ik, :])
                target = converted[iorb, iorb, ispin, ispin, ik, :]
                dynamic = target - c0
                scale = float(max(np.max(np.abs(target)), 1.0))
                projected_dynamic, broken = ref_projector.project(
                    dynamic,
                    moments={"m1": -float(c1), "m2": -float(c2)},
                    scale=scale,
                )
                if broken:
                    raise RuntimeError(
                        f"causal_boson.py projection failed for channel "
                        f"iorb={iorb}, spin={ispin}, k={ik}"
                    )
                out[iorb, iorb, ispin, ispin, ik, :] = projected_dynamic + c0

    return np.asfortranarray(out)


def qassemble_scalar_projection(
    owner: P | W,
    uniform_data: np.ndarray,
    *,
    coefficient_sign: int,
    reflection_symmetry: bool,
    constraint_tol: float,
) -> np.ndarray:
    return owner.CausalProjection(
        uniform_data,
        grid="uniform",
        coefficient_sign=coefficient_sign,
        reflection_symmetry=reflection_symmetry,
        constraint_tol=constraint_tol,
    )


def print_difference(label: str, qassemble: np.ndarray, reference: np.ndarray) -> None:
    diff = qassemble - reference
    denom = max(float(np.linalg.norm(reference)), 1.0e-30)
    print(
        f"{label}: max|QAssemble-reference|={np.max(np.abs(diff)):.3e}, "
        f"rel={np.linalg.norm(diff) / denom:.3e}"
    )


def plot_channel(
    label: str,
    dlr: DLR,
    original_dlr: np.ndarray,
    noisy_uniform: np.ndarray,
    qassemble: np.ndarray,
    reference: np.ndarray,
    *,
    save_dir: Path | None,
) -> None:
    nu_uniform = dlr.MatsubaraBosonUniform()
    fig = plt.figure(figsize=(12, 8))
    for panel, part, part_label in (
        (1, np.real, "Real"),
        (2, np.imag, "Imag"),
    ):
        plt.subplot(2, 1, panel)
        # plt.plot(dlr.nu, part(original_dlr), "o-", label="Original DLR")
        # plt.plot(nu_uniform, part(noisy_uniform), ".-", alpha=0.45, label="Noisy uniform")
        plt.plot(dlr.nu, part(qassemble), "x-", label="QAssemble causal")
        plt.plot(dlr.nu, part(reference), "+--", label="causal_boson.py")
        plt.xlabel(r"$\nu_n$")
        plt.ylabel(f"{part_label} {label}")
        plt.title(f"{label} causal projection comparison ({part_label})")
        plt.legend()
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{label.lower()}_causal_projection.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_CAUSAL_BOSON)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--noise", type=float, default=1.0e-4)
    parser.add_argument("--constraint-tol", type=float, default=1.0e-7)
    parser.add_argument("--p-coefficient-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--w-coefficient-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args()

    causal_boson = load_reference_causal_boson(args.reference)
    crystal, dlr, hamtb = single_band_hubbard()
    g0 = G0(crystal=crystal, dlr=dlr, hamtb=hamtb, hdf5file=None)
    p = P(crystal=crystal, dlr=dlr, green=g0.rt, hdf5file=None)
    vbare = single_band_vbare(crystal)
    w = W(crystal=crystal, dlr=dlr, pol=p.kf, vbare=vbare, hdf5file=None)

    p_uniform = boson_lat_dlr_to_uniform(dlr, p.kf)
    p_noisy = add_low_frequency_noise(
        p_uniform,
        dlr.MatsubaraBosonUniform(),
        cutoff=dlr.cutoff,
        norm=args.noise,
        seed=11,
    )
    p_qassemble = qassemble_scalar_projection(
        p,
        p_noisy,
        coefficient_sign=args.p_coefficient_sign,
        reflection_symmetry=True,
        constraint_tol=args.constraint_tol,
    )
    p_reference = reference_causal_projection(
        p,
        p_noisy,
        causal_boson,
        coefficient_sign=args.p_coefficient_sign,
        reflection_symmetry=True,
        constraint_tol=args.constraint_tol,
    )
    print_difference("P", p_qassemble, p_reference)

    wc_uniform = boson_lat_dlr_to_uniform(dlr, w.ckf)
    w_uniform = wc_uniform + w.vbare.k[..., np.newaxis]
    w_noisy = add_low_frequency_noise(
        w_uniform,
        dlr.MatsubaraBosonUniform(),
        cutoff=dlr.cutoff,
        norm=args.noise,
        seed=17,
    )
    w_qassemble = qassemble_scalar_projection(
        w,
        w_noisy,
        coefficient_sign=args.w_coefficient_sign,
        reflection_symmetry=True,
        constraint_tol=args.constraint_tol,
    )
    w_reference = reference_causal_projection(
        w,
        w_noisy,
        causal_boson,
        coefficient_sign=args.w_coefficient_sign,
        reflection_symmetry=True,
        constraint_tol=args.constraint_tol,
    )
    print_difference("W", w_qassemble, w_reference)

    plot_channel(
        "P",
        dlr,
        p.kf[0, 0, 0, 0, 0, :],
        p_noisy[0, 0, 0, 0, 0, :],
        p_qassemble[0, 0, 0, 0, 0, :],
        p_reference[0, 0, 0, 0, 0, :],
        save_dir=args.save_dir,
    )
    plot_channel(
        "W",
        dlr,
        w.kf[0, 0, 0, 0, 0, :],
        w_noisy[0, 0, 0, 0, 0, :],
        w_qassemble[0, 0, 0, 0, 0, :],
        w_reference[0, 0, 0, 0, 0, :],
        save_dir=args.save_dir,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
