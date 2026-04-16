from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .BasisIndex import BasisIndex


class Projector(object):
    """Build impurity projectors on a per-problem (`nproblem`) basis.

    Notes:
        - ``probspace`` stores atom-site indices per equivalent space.
        - ``probindex`` stores contiguous local-space indices used internally
          for impurity/local array conversions.
        - ``fprojector`` / ``bprojector`` are dense, padded arrays kept for
          compatibility with existing solvers.
        - ``fprojector_prob`` / ``bprojector_prob`` are per-problem compact
          projectors with variable orbital dimension.
    """

    def __init__(self, basisindex: "BasisIndex", impdict: dict = None):
        self.basisindex = basisindex
        self.impdict = impdict

        self.probspace = {}
        self.probindex = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.fprojector = {}
        self.bprojector = {}
        self.fprojector_prob = None
        self.bprojector_prob = None

        if impdict is not None:
            self.Build(impdict)

    def _validate_impdict(self, impdict: dict) -> None:
        if impdict is None:
            raise ValueError("impdict cannot be None")

        for key, spaces in impdict.items():
            if len(spaces) == 0:
                raise ValueError(f"impdict['{key}'] must contain at least one space")

            for orblist in spaces:
                if len(orblist) == 0:
                    raise ValueError(
                        f"impdict['{key}'] contains an empty orbital list"
                    )
                atom = orblist[0][0]
                for orb in orblist:
                    if atom != orb[0]:
                        raise ValueError(
                            "Different atoms are involved in the same space"
                        )

    def Build(self, impdict: dict = None) -> None:
        if impdict is not None:
            self.impdict = impdict
        if self.impdict is None:
            raise ValueError("impdict is not set")

        self._validate_impdict(self.impdict)

        ns = self.basisindex.ns
        nproblem = len(self.impdict)
        nspace = 0

        probspace = {}
        probindex = {}
        fimpdict = {}
        bimpdict = {}
        forbc = 0
        borbc = 0

        for key, val in self.impdict.items():
            iprob = int(key) - 1
            if iprob < 0 or iprob >= nproblem:
                raise ValueError(
                    f"impurity key '{key}' maps to invalid problem index {iprob}"
                )

            # probspace: atom-site labels aligned with equivalent-space order.
            probspace[key] = [int(orblist[0][0]) for orblist in val]

            # probindex: contiguous equivalent-space index for local array axis.
            probindex[key] = [nspace + i for i in range(len(val))]
            nspace += len(val)

            fimpdict[key] = []
            bimpdict[key] = []

            ref_len = None
            for orblist in val:
                f_orbs = []
                for orb in orblist:
                    find = self.basisindex.FIndex(orb)
                    f_orbs.append(find)
                fimpdict[key].append(f_orbs)
                forbc = max(forbc, len(f_orbs))

                if ref_len is None:
                    ref_len = len(f_orbs)
                elif len(f_orbs) != ref_len:
                    raise ValueError(
                        f"All equivalent spaces in impurity '{key}' must have the same orbital count"
                    )

                b_orbs = []
                for iorb in f_orbs:
                    for jorb in f_orbs:
                        a, _ = self.basisindex.FAtomOrb(iorb)
                        b, _ = self.basisindex.FAtomOrb(jorb)
                        if a == b:
                            bind = self.basisindex.bbasis[iorb, jorb] - 1  # 1-based -> 0-based
                            b_orbs.append(bind)
                bimpdict[key].append(b_orbs)
                borbc = max(borbc, len(b_orbs))

        # Compact projectors per impurity problem: variable second dimension.
        fprojector = {}
        bprojector = {}
        for key in self.impdict.keys():
            fcols = len(fimpdict[key][0])
            bcols = len(bimpdict[key][0])
            fproj = np.zeros((len(self.basisindex.find), fcols, ns), dtype=float)
            bproj = np.zeros((len(self.basisindex.bind), bcols, ns), dtype=float)

            rep_f_orbs = fimpdict[key][0]
            rep_b_orbs = bimpdict[key][0]
            for js in range(ns):
                for col, ind in enumerate(rep_f_orbs):
                    fproj[ind, col, js] = 1.0
                for col, ind in enumerate(rep_b_orbs):
                    bproj[ind, col, js] = 1.0

            fprojector[key] = fproj
            bprojector[key] = bproj

        fprojector_prob = np.zeros((len(self.basisindex.find), forbc, ns, nproblem), dtype=float)
        bprojector_prob = np.zeros((len(self.basisindex.bind), borbc, ns, nproblem), dtype=float)

        for js in range(ns):
            for key in probspace.keys():
                iprob = int(key) - 1
                rep_f_orbs = fimpdict[key][0]
                rep_b_orbs = bimpdict[key][0]
                for col, ind in enumerate(rep_f_orbs):
                    fprojector_prob[ind, col, js, iprob] = 1.0
                for col, ind in enumerate(rep_b_orbs):
                    bprojector_prob[ind, col, js, iprob] = 1.0

        self.probspace = probspace
        self.probindex = probindex
        self.fimpdict = fimpdict
        self.bimpdict = bimpdict
        self.fprojector = fprojector
        self.bprojector = bprojector
        self.fprojector_prob = fprojector_prob
        self.bprojector_prob = bprojector_prob

        return None
