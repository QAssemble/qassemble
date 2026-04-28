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

    def __init__(self, basisindex: "BasisIndex", impdict: dict = None, equiv : dict = None):
        self.basisindex = basisindex
        self.impdict = impdict
        self.equiv = equiv

        self.probspace = {}
        self.probindex = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.bpair2local = {}
        self.blocal2pair = {}
        self.findex2local = {}
        self.bindex2local = {}
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

    def _require_problem(self, key: str) -> str:
        key = str(key)
        if key not in self.fimpdict:
            raise KeyError(f"Unknown impurity problem key '{key}'")
        return key

    def _require_space(self, key: str, ispace: int) -> str:
        key = self._require_problem(key)
        if ispace < 0 or ispace >= len(self.fimpdict[key]):
            raise IndexError(
                f"Space index {ispace} is out of range for impurity '{key}'"
            )
        return key

    def _require_local_orb(self, key: str, ispace: int, iorb: int) -> str:
        key = self._require_space(key, ispace)
        if iorb < 0 or iorb >= len(self.fimpdict[key][ispace]):
            raise IndexError(
                f"Orbital index {iorb} is out of range for impurity '{key}', space {ispace}"
            )
        return key

    def _require_local_borb(self, key: str, ispace: int, iborb: int) -> str:
        key = self._require_space(key, ispace)
        if iborb < 0 or iborb >= len(self.bimpdict[key][ispace]):
            raise IndexError(
                f"Boson index {iborb} is out of range for impurity '{key}', space {ispace}"
            )
        return key

    def Prob2FIndex(self, key: str, iorb: int, ispace: int = 0) -> int:
        """Map a local impurity fermion orbital to the global fermion index.

        Args:
            key (str): Impurity problem key.
            iorb (int): Local fermion orbital index within the selected space.
            ispace (int): Equivalent-space index within the impurity problem.

        Returns:
            int: Global fermion index in ``basisindex.find``.
        """
        key = self._require_local_orb(key, ispace, iorb)
        return self.fimpdict[key][ispace][iorb]

    def Prob2SiteOrb(self, key: str, iorb: int, ispace: int = 0) -> tuple:
        """Map a local impurity fermion orbital to ``(site, orbital)`` labels.

        Args:
            key (str): Impurity problem key.
            iorb (int): Local fermion orbital index within the selected space.
            ispace (int): Equivalent-space index within the impurity problem.

        Returns:
            tuple: ``(site, orbital)`` corresponding to the local impurity orbital.
        """
        findex = self.Prob2FIndex(key, iorb, ispace=ispace)
        site, morb = self.basisindex.FAtomOrb(findex)
        return site, morb

    def FIndex2Prob(self, findex: int) -> tuple:
        """Map a global fermion index back to impurity-local coordinates.

        Args:
            findex (int): Global fermion index.

        Returns:
            tuple: ``(key, ispace, iorb)`` for the matching impurity-local orbital.
        """
        findex = int(findex)
        if findex not in self.findex2local:
            raise ValueError(
                f"Global fermion index {findex} is not part of any impurity space"
            )
        return self.findex2local[findex]

    def Prob2BIndex(
        self,
        key: str,
        iorbc: int = None,
        ispace: int = 0,
        iorb: int = None,
        jorb: int = None,
    ) -> int:
        """Map an impurity-local boson label or orbital pair to a global boson index.

        Args:
            key (str): Impurity problem key.
            iorbc (int): Local boson flat index within the selected space.
            ispace (int): Equivalent-space index within the impurity problem.
            iorb (int): First local fermion orbital index for pair-based lookup.
            jorb (int): Second local fermion orbital index for pair-based lookup.

        Returns:
            int: Global boson index in ``basisindex.bind``.

        Notes:
            - Provide either ``iborb`` directly, or both ``iorb`` and ``jorb``.
            - Pair-based lookup uses the fermion pair ordering of the selected
              impurity space.
        """
        key = self._require_space(key, ispace)

        if iorbc is not None:
            key = self._require_local_borb(key, ispace, iorbc)
            return self.bimpdict[key][ispace][iorbc]

        if iorb is None or jorb is None:
            raise ValueError(
                "Either iorbc or both iorb and jorb must be provided"
            )

        key = self._require_local_orb(key, ispace, iorb)
        key = self._require_local_orb(key, ispace, jorb)
        ifind = self.fimpdict[key][ispace][iorb]
        jfind = self.fimpdict[key][ispace][jorb]
        return self.basisindex.bbasis[ifind, jfind] - 1

    def ProbFPair2Borb(self, key: str, iorb: int, jorb: int, ispace: int = 0) -> int:
        """Map a local fermion-orbital pair to a local boson index.

        Args:
            key (str): Impurity problem key.
            iorb (int): First local fermion orbital index.
            jorb (int): Second local fermion orbital index.
            ispace (int): Equivalent-space index within the impurity problem.

        Returns:
            int: Local boson index (problem dimension) in ``bimpdict[key][ispace]``.
        """
        key = self._require_local_orb(key, ispace, iorb)
        key = self._require_local_orb(key, ispace, jorb)

        pair = (int(iorb), int(jorb))
        mapping = self.bpair2local[key][ispace]
        if pair not in mapping:
            raise ValueError(
                f"Local fermion pair {pair} is not mapped in impurity '{key}', space {ispace}"
            )
        return mapping[pair]

    def ProbBorb2FPair(self, key: str, iorbc: int, ispace: int = 0) -> tuple:
        """Map a local boson index to the corresponding local fermion-orbital pair.

        Args:
            key (str): Impurity problem key.
            iorbc (int): Local boson index.
            ispace (int): Equivalent-space index within the impurity problem.

        Returns:
            tuple: ``(iorb, jorb)`` local fermion indices in problem dimension.
        """
        key = self._require_local_borb(key, ispace, iorbc)
        return self.blocal2pair[key][ispace][int(iorbc)]

    def Prob2SiteOrbs(
        self,
        key: str,
        iorbc: int = None,
        ispace: int = 0,
        iorb: int = None,
        jorb: int = None,
    ) -> tuple:
        """Map an impurity-local boson label or orbital pair to ``(site, pair)``.

        Args:
            key (str): Impurity problem key.
            iorbc (int): Local boson flat index within the selected space.
            ispace (int): Equivalent-space index within the impurity problem.
            iorb (int): First local fermion orbital index for pair-based lookup.
            jorb (int): Second local fermion orbital index for pair-based lookup.

        Returns:
            tuple: ``(site, [m1, m2])`` corresponding to the boson index.
        """
        bindex = self.Prob2BIndex(
            key, iorbc=iorbc, ispace=ispace, iorb=iorb, jorb=jorb
        )
        site, morbs = self.basisindex.BAtomOrb(bindex)
        return site, morbs

    def BIndex2Prob(self, bindex: int) -> tuple:
        """Map a global boson index back to impurity-local coordinates.

        Args:
            bindex (int): Global boson index.

        Returns:
            tuple: ``(key, ispace, iorbc, (iorb, jorb))`` for the matching
            impurity-local boson entry.
        """
        bindex = int(bindex)
        if bindex not in self.bindex2local:
            raise ValueError(
                f"Global boson index {bindex} is not part of any impurity space"
            )
        return self.bindex2local[bindex]

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
        bpair2local = {}
        blocal2pair = {}
        findex2local = {}
        bindex2local = {}
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
            bpair2local[key] = []
            blocal2pair[key] = []

            ref_len = None
            for ispace, orblist in enumerate(val):
                f_orbs = []
                for iorb, orb in enumerate(orblist):
                    find = self.basisindex.FIndex(orb)
                    if find in findex2local:
                        prev_key, prev_space, prev_orb = findex2local[find]
                        raise ValueError(
                            "Duplicate fermion orbital mapping detected: "
                            f"findex {find} appears in impurity '{prev_key}' space {prev_space} orbital {prev_orb} "
                            f"and impurity '{key}' space {ispace} orbital {iorb}"
                        )
                    findex2local[find] = (key, ispace, iorb)
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
                pair2local = {}
                local2pair = {}
                for iborb, iorb in enumerate(f_orbs):
                    for jborb, jorb in enumerate(f_orbs):
                        a, _ = self.basisindex.FAtomOrb(iorb)
                        b, _ = self.basisindex.FAtomOrb(jorb)
                        if a == b:
                            bind = self.basisindex.bbasis[iorb, jorb] - 1  # 1-based -> 0-based
                            local_borb = len(b_orbs)
                            if bind in bindex2local:
                                prev_key, prev_space, prev_iborb, prev_pair = bindex2local[bind]
                                raise ValueError(
                                    "Duplicate boson orbital mapping detected: "
                                    f"bindex {bind} appears in impurity '{prev_key}' space {prev_space} local boson {prev_iborb} "
                                    f"(pair {prev_pair}) and impurity '{key}' space {ispace} "
                                    f"(pair ({iborb}, {jborb}))"
                                )
                            bindex2local[bind] = (key, ispace, local_borb, (iborb, jborb))
                            b_orbs.append(bind)
                            pair2local[(iborb, jborb)] = local_borb
                            local2pair[local_borb] = (iborb, jborb)
                bimpdict[key].append(b_orbs)
                bpair2local[key].append(pair2local)
                blocal2pair[key].append(local2pair)
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
        self.bpair2local = bpair2local
        self.blocal2pair = blocal2pair
        self.findex2local = findex2local
        self.bindex2local = bindex2local
        self.fprojector = fprojector
        self.bprojector = bprojector
        self.fprojector_prob = fprojector_prob
        self.bprojector_prob = bprojector_prob

        return None
