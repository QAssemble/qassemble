import numpy as np
import sys
import itertools
import copy
from .utility.Common import Common


class BasisIndex(object):
    """Manages quantum basis indexing for fermionic and bosonic orbitals.

    This class constructs and manages indices for fermionic and bosonic orbitals,
    and provides methods to map between different basis representations including
    tensor dimension transformations.
    """

    def __init__(self, orboption: dict, ns: int):
        """Initialize the BasisIndex object.

        Args:
            orboption (dict): Mapping from atom index to number of orbitals.
            ns (int): Number of spin degrees of freedom.
        """
        self.ns = ns
        self.find = {}
        self.bind = {}
        self.full = {}
        self.c2b = None
        self.forb2atom = None
        self.borb2atom = None
        self.probspace = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.fprojector = None
        self.bprojector = None

        self.SetBasisIndex(orboption)
        self.pbasis = np.zeros((len(self.find), len(self.find)), dtype=int)
        self.bbasis = np.zeros((len(self.find), len(self.find)), dtype=int)
        self.Boson2Fermion()
        self.SetFullBasis()
        self.Boson2Full()

    def SetBasisIndex(self, orboption: dict) -> dict:
        """Set up fermion and boson orbital mappings from orbital options.

        Args:
            orboption (dict): Mapping from atom index to number of orbitals.

        Returns:
            None
        """
        for key, val in orboption.items():
            find = []
            bind = []
            orblist = list(range(val))

            for m1 in range(val):
                find.append([key,m1])
            for m2, m1 in itertools.product(orblist,orblist):
                bind.append([key,[m1,m2]])
            forb = len(self.find)
            borb = len(self.bind)

            ii = 0
            for iorb in range(forb,forb+val):
                self.find[iorb] = find[ii]
                ii += 1
            jj = 0
            for iorb in range(borb,borb+val**2):
                self.bind[iorb] = bind[jj]
                jj+=1

            self.forb2atom = np.empty(len(self.find), dtype=np.int32)
            self.borb2atom = np.empty(len(self.bind), dtype=np.int32)

            for iorb in range(len(self.find)):
                a, _ = self.FAtomOrb(iorb)
                self.forb2atom[iorb] = a

            for iorb in range(len(self.bind)):
                a, _ = self.BAtomOrb(iorb)
                self.borb2atom[iorb] = a

        return None

    def FAtomOrb(self, key: int) -> list:
        """Get atom and orbital indices for a given fermion composite index.

        Args:
            key (int): Fermion composite index.

        Returns:
            list[int]: [atom_index, orbital_index] corresponding to the composite index.
        """
        return self.find[key]

    def FIndex(self, val: list) -> int:
        """Get the fermion composite index for given atom and orbital indices.

        Args:
            val (list[int]): [atom_index, orbital_index].

        Returns:
            int: Corresponding fermion composite index.
        """

        for key, value in self.find.items():
            if value == val:
                return key

    def BAtomOrb(self, key: int) -> list:
        """Get atom and orbital pair for a given boson composite index.

        Args:
            key (int): Boson composite index.

        Returns:
            list: [atom_index, [orbital1_index, orbital2_index]].
        """
        return self.bind[key]

    def BIndex(self, val: list) -> int:
        """Get the boson composite index for given atom and orbital pair.

        Args:
            val (list): [atom_index, [orbital1_index, orbital2_index]].

        Returns:
            int: Corresponding boson composite index.
        """
        for key, value in self.bind.items():
            if val==value:
                return key

    def Boson2Fermion(self):
        """Populate bbasis mapping from fermion indices to boson composite indices.

        Returns:
            None
        """
        norbc = len(self.find)
        bbasis = np.zeros((norbc,norbc),dtype=int)
        for jorbc in range(norbc):
            for iorbc in range(norbc):
                [a,m] = self.FAtomOrb(iorbc)
                [ap,mp] = self.FAtomOrb(jorbc)
                if (a==ap):
                    iorb = self.BIndex([a,[m,mp]])
                    bbasis[iorbc,jorbc] = iorb

        self.bbasis = bbasis

        return None

    def Boson2Full(self):

        norb = len(self.bind)
        c2b = np.zeros((norb),dtype=int)

        for iorb in range(norb):
            [a,[m,mp]] = self.BAtomOrb(iorb)
            iorbc = self.FIndex([a,m])
            jorbc = self.FIndex([a,mp])
            ind = self.pbasis[iorbc,jorbc]
            c2b[iorb] = ind

        self.c2b = c2b

    def SetFullBasis(self):

        norbc = len(self.find)
        full = {}
        pbasis = np.zeros((norbc,norbc),dtype=int)

        for jorbc in range(norbc):
            for iorbc in range(norbc):
                (a,m1) = self.FAtomOrb(iorbc)
                (b,m2) = self.FAtomOrb(jorbc)
                nn = [iorbc,jorbc]
                ind, nn = Common.Indexing(norbc*norbc,2,[norbc,norbc],1,0,nn)
                full[ind] = [[a,m1],[b,m2]]
                pbasis[iorbc,jorbc] = ind

        self.pbasis = copy.deepcopy(pbasis)
        self.full = copy.deepcopy(full)

        return None

    def FullIndex(self, val: list):
        """Get the full composite index for given orbital pairing.

        Args:
            val (list): [[atom1, orb1], [atom2, orb2]].

        Returns:
            int: Composite full index.
        """

        for k, v in self.full.items():
            if v == val:
                return k

    def FullAtomOrb(self, ind: int):
        """Get atom and orbital indices from full composite index.

        Args:
            ind (int): Full composite index.

        Returns:
            list: [[atom1, orb1], [atom2, orb2]].
        """
        return self.full[ind]

    def Composite2Fermion(self):
        """Generate mapping from composite indices to fermion index pairs.

        Returns:
            None
        """
        norbc = len(self.find)
        norb = norbc*norbc
        c2f = []

        for iorbc in range(norbc):
            for jorbc in range(norbc):
                nn1 = [iorbc,jorbc]
                iorb, nn1 = Common.Indexing(norb,2,[norbc,norbc],1,0,nn1)
                c2f.append([iorbc,jorbc])
        self.c2f = c2f

    def Composite2Boson(self):
        """Generate mapping from composite indices to boson composite indices.

        Returns:
            None
        """

        norbc = len(self.find)
        ndim = norbc*norbc
        c2b = []

        for ind in range(ndim):
            nn1 = [0]*2
            ind,[iorbc,jorbc] = Common.Indexing(ndim,2,[norbc,norbc],0,ind,nn1)
            [a,m1] = self.FAtomOrb(iorbc)
            [a_p,m2] = self.FAtomOrb(jorbc)
            if a==a_p:
                borb = self.BIndex([a,[m1,m2]])
                if borb is not None:
                    c2b.append([borb,ind])
        self.c2b = c2b

    def Composite2OrbSpin(self, mat: np.ndarray):
        """Reshape a composite matrix into orbital-spin representation.

        Args:
            mat (np.ndarray): Composite matrix of shape (norb*ns, norb*ns).

        Returns:
            np.ndarray: Array of shape (norb, norb, ns, ns).
        """

        norb = len(self.full)
        ndim = mat.shape[0]
        ns = self.ns

        idx = np.arange(ndim)
        iorb_arr = idx % norb
        js_arr = idx // norb

        matout = np.zeros((norb, norb, ns, ns), dtype=np.complex128, order='F')

        matout[iorb_arr[:, None], iorb_arr[None, :],
               js_arr[:, None], js_arr[None, :]] = mat

        return matout

    def OrbSpin2Composite(self, mat: np.ndarray):
        """Reshape an orbital-spin matrix into composite matrix form.

        Args:
            mat (np.ndarray): Array of shape (norb, norb, ns, ns).

        Returns:
            np.ndarray: Composite matrix of shape (norb*ns, norb*ns).
        """

        norb = mat.shape[0]
        ns = self.ns
        ndim = norb * ns

        idx = np.arange(ndim)
        iorb_arr = idx % norb
        js_arr = idx // norb

        matout = mat[iorb_arr[:, None], iorb_arr[None, :],
                     js_arr[:, None], js_arr[None, :]]

        return np.array(matout, dtype=np.complex128, order='F')

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:
        """Convert a 4-index tensor to 2-index matrix in boson basis.

        Args:
            matin (np.ndarray): 4D array of shape (norbc, norbc, norbc, norbc).

        Returns:
            np.ndarray: 2D array of shape (norb, norb).
        """

        norb = len(self.bind)
        norbc = len(self.find)
        matout = np.zeros((norb,norb),dtype=np.complex64)

        for l, k, j, i in itertools.product(range(norbc), repeat=4):
            iorb = self.bbasis[i, l]
            jorb = self.bbasis[j, k]
            matout[iorb, jorb] = matin[i, j, k, l]

        return matout

    def Double2Quad(self, matin : np.ndarray) -> np.ndarray:
        """Convert a 2-index matrix in boson basis to a 4-index tensor.

        Args:
            matin (np.ndarray): 2D array of shape (norb, norb).

        Returns:
            np.ndarray: 4D array of shape (norbc, norbc, norbc, norbc).
        """

        norbc = len(self.find)
        norb = len(self.bind)

        matout = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        for l, k, j, i in itertools.product(range(norbc), repeat=4):
            iorb = self.bbasis[i, l]
            jorb = self.bbasis[j, k]
            matout[i, j, k, l] = matin[iorb, jorb]

        return matout

    def Full2Quad(self, matin : np.ndarray) -> np.ndarray:
        """Convert a full composite matrix to a 4-index tensor.

        Args:
            matin (np.ndarray): 2D array of shape (n^2, n^2).

        Returns:
            np.ndarray: 4D array of shape (n, n, n, n).
        """

        norbc = len(self.find)

        matout = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        for l, k, j, i in itertools.product(range(norbc), repeat=4):
            iorb = self.pbasis[i, l]
            jorb = self.pbasis[j, k]
            matout[i, j, k, l] = matin[iorb, jorb]


        return matout

    def Quad2Full(self, mat: np.ndarray) -> np.ndarray:
        """Convert a 4-index tensor to a full composite matrix.

        Args:
            mat (np.ndarray): 4D array of shape (n, n, n, n).

        Returns:
            np.ndarray: 2D array of shape (n^2, n^2).
        """

        norbc = len(self.find)

        matret = np.zeros((norbc**2,norbc**2))

        for lorbc in range(norbc):
            for korbc in range(norbc):
                for jorbc in range(norbc):
                    for iorbc in range(norbc):
                        iorb = self.pbasis[iorbc,lorbc]
                        jorb = self.pbasis[jorbc,korbc]
                        matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

        return matret

    def Full2Double(self, mat: np.ndarray) -> np.ndarray:
        """Convert a full composite matrix to a boson basis 2-index matrix.

        Args:
            mat (np.ndarray): 2D array of shape (n^2, n^2).

        Returns:
            np.ndarray: 2D array of shape (norb, norb).
        """

        c2b = np.asarray(self.c2b)

        matret = mat[np.ix_(c2b, c2b)]

        return np.array(matret, dtype=np.complex128, order='F')

    def Double2Full(self, mat: np.ndarray) -> np.ndarray:
        """Convert a boson basis 2-index matrix to a full composite matrix.

        Args:
            mat (np.ndarray): 2D array of shape (norb, norb).

        Returns:
            np.ndarray: 2D array of shape (n^2, n^2).
        """

        nind = len(self.find)**2

        c2b = np.asarray(self.c2b, dtype=np.int64)
        matret = np.zeros((nind,nind),dtype=np.complex128, order='F')

        rhs = np.asarray(mat, dtype=np.complex128, order='F')

        matret[np.ix_(c2b, c2b)] = rhs

        return matret

    def Projector(self, impdict: dict):
        """Generate fermion and boson projectors for impurity calculations.

        Args:
            impdict (dict): Mapping of impurity labels to lists of orbital index lists.
            e.g.
            impdict = {
                '1'  : [[[0, 0], [0, 1]],[[1, 0]]]
                }

        Returns:
            None
        """

        nspace = 0
        forbc = 0
        borbc = 0
        ns = self.ns
        probspace = {}
        fimpdict = {}
        bimpdict = {}

        for key, val in impdict.items():
            for orblist in val:
                atom = 0
                for orb in orblist:
                    if orb == orblist[0]:
                        atom = orb[0]
                    if atom != orb[0]:
                        print("Different atoms are involved in the same space")
                        sys.exit()
            probspace[key] = [nspace+i for i in range(len(val))]
            nspace += len(val)

        self.probspace = probspace

        for key, val in impdict.items():
            fimpdict[key] = []
            for orblist in val:
                templist = []
                for orb in orblist:
                    find = self.FIndex(orb)
                    templist.append(find)
                fimpdict[key].append(templist)
        self.fimpdict = fimpdict
        for val in fimpdict.values():
            for orb in val:
                if len(orb) > forbc:
                    forbc = len(orb)
        for key, val in fimpdict.items():
            bimpdict[key] = []
            for orb in val:
                templist = []
                for iorb in orb:
                    for jorb in orb:
                        [a,m1] = self.FAtomOrb(iorb)
                        [b,m2] = self.FAtomOrb(jorb)
                        if a==b:
                            bind = self.bbasis[iorb, jorb]
                            templist.append(bind)
                bimpdict[key].append(templist)
        for val in bimpdict.values():
            for orb in val:
                if len(orb)>borbc:
                    borbc = len(orb)
        self.bimpdict = bimpdict
        fprojector = np.zeros((len(self.find),forbc,ns,nspace),dtype=float,order='F')
        bprojector = np.zeros((len(self.bind),borbc,ns,nspace),dtype=float,order='F')

        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.fimpdict[key][ii]:
                        fprojector[ind,self.fimpdict[key][ii].index(ind),js,ispace] = 1.0

        for js in range(ns):
            for key, val in probspace.items():
                for ii, ispace in enumerate(val):
                    for ind in self.bimpdict[key][ii]:
                        bprojector[ind,self.bimpdict[key][ii].index(ind),js,ispace] = 1.0

        self.fprojector = fprojector
        self.bprojector = bprojector

        return None

    def MappingBosonFermion(self, iorb):

        [a, [m1, m4]] = self.BAtomOrb(iorb)

        iorbc = self.FIndex([a, m1])
        lorbc = self.FIndex([a, m4])

        return iorbc, lorbc
