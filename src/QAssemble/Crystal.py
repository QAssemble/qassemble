import numpy as np
import sys
import itertools

import copy
from .utility.Common import Common

# Ask to professor for change variables
class Crystal(object):
    """Handles lattice geometry, orbitals, and basis indexing for quantum assembly calculations.

    This class constructs indices and vectors for fermionic and bosonic orbitals,
    k-point grids, real-space vectors, and provides methods to map between different basis representations.
    """
    def __init__(self, cry: dict = None):
        """Initialize the Crystal object.

        Args:
            cry (dict): Dictionary containing crystal parameters. Expected keys:
                RVec (list of list[float]): Real-space lattice vectors.
                Basis (list): List of [position, orbital_count] entries.
                CorF (str): 'C' for Cartesian coords input, 'F' for fractional coords.
                NSpin (int): Number of spin degrees of freedom.
                SOC (bool): Spin-orbit coupling flag.
                NElec (float): Number of electrons.
                KGrid (list of int]): Grid dimensions for k-point sampling.
        """

        Rvec = cry['RVec']
        Basis = cry['Basis']
        # CorF = cry['CorF']
        CorF = cry.get('CorF', 'F')
        Nspin = cry['NSpin']
        # SOC = cry['SOC']
        SOC = cry.get('SOC', False)
        Nelec = cry['NElec']
        KGrid = cry['KGrid']
        self.avec = np.array(Rvec,dtype=float)
        pos = []
        orboption = {}
        for i, ii in enumerate(Basis):
            pos.append(ii[0])
            orboption[i] = ii[1]
        pos = np.array(pos)
        if CorF == "C":
            self.basisc = pos
            self.basisf = np.dot(self.basisc,np.linalg.inv(self.avec))
        elif CorF == "F":
            self.basisf = pos
            self.basisc = np.dot(self.basisf,self.avec)

        self.ns = Nspin
        self.soc = SOC
        self.nume = Nelec*(Nspin/2)
        self.bvec = np.zeros((3,3))
        self.vol=np.dot(np.cross(self.avec[:,0], self.avec[:,1]), self.avec[:,2])
        self.bvec[:,0]=2*np.pi*np.cross(self.avec[:,1], self.avec[:,2])/self.vol
        self.bvec[:,1]=2*np.pi*np.cross(self.avec[:,2], self.avec[:,0])/self.vol
        self.bvec[:,2]=2*np.pi*np.cross(self.avec[:,0], self.avec[:,1])/self.vol
        
        
        self.rkgrid = KGrid
        self.nk = KGrid[0]*KGrid[1]*KGrid[2]
        kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=KGrid[2],endpoint=False),np.linspace(0,1,num=KGrid[1],endpoint=False),np.linspace(0,1,num=KGrid[0],endpoint=False))))
        kpoint=np.fliplr(kpoint_temp)
        self.kpoint = kpoint

        svec = np.zeros((3,3),dtype=np.float64,order='F')
        svec[0] = KGrid[0]*self.avec[0]
        svec[1] = KGrid[1]*self.avec[1]
        svec[2] = KGrid[2]*self.avec[2]
        self.svec = svec

        self.rvec = None
        self.rind = None
        self.kpath = None
        self.kdist = None
        self.knode = None
        self.kind = {}
        self.K2K3D()

        self.find = {}
        self.bind = {}
        self.full = {}
        # 
        # self.b2f = None
        self.c2b = None
        self.probspace = {}
        self.fimpdict = {}
        self.bimpdict = {}
        self.fprojector = None
        self.bprojector = None

        self.mappingidx = []
        self.mappingkp = []
#       templist = []
#       for key, val in orboption.items():
#           templist.append([key,val])
        self.orboption = orboption


        self.SetBasisIndex(orboption)
        self.pbasis = np.zeros((len(self.find),len(self.find)),dtype=int)
        self.bbasis = np.zeros((len(self.find),len(self.find)),dtype=int)
        self.Boson2Fermion()
        self.SetFullBasis()
        self.Boson2Full()
        self.RVec()

        return None

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

        return None


#       for option in orboption:
#           find = []
#           bind = []
#           orblist = list(range(option[1]))

#           for m1 in range(option[1]):
#               find.append([option[0],m1])
#           for m2, m1 in itertools.product(orblist,orblist):
#               bind.append([option[0],[m1,m2]])

#           forb = len(self.find)
#           borb = len(self.bind)
#           ii = 0
#           for iorb in range(forb,forb+option[1]):
#               self.find[iorb] = find[ii]
#               ii +=1
#           ii = 0
#           for iorb in range(borb,borb+option[1]**2):
#               self.bind[iorb] = bind[ii]
#               ii +=1
                # self.Boson2Fermion(iorb)
            # self.Composite2Boson()
            # self.Composite2Fermion()


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

        norb = len(self.bind)
        ns = self.ns
        matout = np.zeros((norb,norb,ns,ns),dtype=np.complex64,order='F')
        ndim = mat.shape[0]

        for ind1 in range(ndim):
            nn1 = [0]*2
            ind1, [iorb,js] = Common(ndim,2,[norb,ns],0,ind1,nn1)
            for ind2 in range(ndim):
                nn2 = [0]*2
                ind2, [jorb,ks] = Common.Indexing(ndim,2,[norb,ns],0,ind2,nn2)
                matout[iorb,jorb,js,ks] = mat[ind1,ind2]

        return matout

    def OrbSpin2Composite(self, mat: np.ndarray):
        """Reshape an orbital-spin matrix into composite matrix form.

        Args:
            mat (np.ndarray): Array of shape (norb, norb, ns, ns).

        Returns:
            np.ndarray: Composite matrix of shape (norb*ns, norb*ns).
        """

        norb = mat.shape[0]
        ns = mat.shape[2]
        matout = np.zeros((norb*ns,norb*ns),dtype=np.complex64,order='F')

        for js in range(ns):
            for iorb in range(norb):
                nn1 = [iorb,js]
                ind1, nn1 = Common.Indexing(norb*ns,2,[norb,ns],1,0,nn1)
                for ks in range(ns):
                    for jorb in range(norb):
                        nn2 = [jorb,ks]
                        ind2, nn2 = Common.Indexing(norb*ns,2,[norb,ns],1,0,nn2)
                        matout[ind1,ind2] = mat[iorb,jorb,js,ks]
        return matout

    def Quad2Double(self, matin: np.ndarray) -> np.ndarray:
        """Convert a 4-index tensor to 2-index matrix in boson basis.

        Args:
            mat (np.ndarray): 4D array of shape (norbc, norbc, norbc, norbc).

        Returns:
            np.ndarray: 2D array of shape (norb, norb).
        """

        norb = len(self.bind)
        norbc = len(self.find)
        matout = np.zeros((norb,norb),dtype=np.complex64)

        # for lorbc in range(norbc):
        #     for korbc in range(norbc):
        #         for jorbc in range(norbc):
        #             for iorbc in range(norbc):
        #                 (a,m1) = self.FAtomOrb(iorbc)
        #                 (ap,m4) = self.FAtomOrb(lorbc)
        #                 (b,m2) = self.FAtomOrb(jorbc)
        #                 (bp,m3) = self.FAtomOrb(korbc)
        #                 if (a==ap)and(b==bp):
        #                     iorb = self.BIndex([a,[m1,m2]])
        #                     jorb = self.BIndex([b,[m2,m3]])
        #                     matret[iorb,jorb] = mat[iorbc,jorbc,korbc,lorbc]

        for l, k, j, i in itertools.product(range(norbc), repeat=4):
            iorb = self.bbasis[i, l]
            jorb = self.bbasis[j, k]
            matout[iorb, jorb] = matin[i, j, k, l]

        return matout

    def Double2Quad(self, matin : np.ndarray) -> np.ndarray:
        """Convert a 2-index matrix in boson basis to a 4-index tensor.

        Args:
            mat (np.ndarray): 2D array of shape (norb, norb).

        Returns:
            np.ndarray: 4D array of shape (norbc, norbc, norbc, norbc).
        """

        norbc = len(self.find)
        norb = len(self.bind)

        matout = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        # for jorb in range(norb):
        #     for iorb in range(norb):
        #         [a,[m1,m4]] = self.BAtomOrb(iorb)
        #         [b,[m2,m3]] = self.BAtomOrb(jorb)
        #         iorbc = self.FIndex([a,m1])
        #         lorbc = self.FIndex([a,m4])
        #         jorbc = self.FIndex([b,m2])
        #         korbc = self.FIndex([b,m3])
        #         matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]

        for l, k, j, i in itertools.product(range(norbc), repeat=4):
            iorb = self.bbasis[i, l]
            jorb = self.bbasis[j, k]
            matout[i, j, k, l] = matin[iorb, jorb]

        return matout

    def Full2Quad(self, matin : np.ndarray) -> np.ndarray:
        """Convert a full composite matrix to a 4-index tensor.

        Args:
            mat (np.ndarray): 2D array of shape (n^2, n^2).

        Returns:
            np.ndarray: 4D array of shape (n, n, n, n).
        """

        norbc = len(self.find)

        matout = np.zeros((norbc,norbc,norbc,norbc),dtype=np.complex64,order='F')

        # for lorbc in range(norbc):
        #     for korbc in range(norbc):
        #         for jorbc in range(norbc):
        #             for iorbc in range(norbc):
        #                 iorb = self.pbasis[iorbc,lorbc]
        #                 jorb = self.pbasis[jorbc,korbc]
        #                 matret[iorbc,jorbc,korbc,lorbc] = mat[iorb,jorb]
        
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

    def Full2Double(self, matin: np.ndarray) -> np.ndarray:
        """Convert a full composite matrix to a boson basis 2-index matrix.

        Args:
            matin (np.ndarray): 2D array of shape (n^2, n^2).

        Returns:
            np.ndarray: 2D array of shape (norb, norb).
        """

        # norb = len(self.bind)

        # matret = np.zeros((norb,norb),dtype=np.complex64,order='F')

        # for jorb in range(norb):
        #     for iorb in range(norb):
        #         ind1 = self.c2b[iorb]
        #         ind2 = self.c2b[jorb]
        #         matret[iorb,jorb] = mat[ind1,ind2]

        c2b = np.asarray(self.c2b, dtype=np.int64)

        matret = matin[np.ix_(c2b, c2b)]

        return np.array(matret, dtype=np.complex128, order='F')

    def Double2Full(self, matin: np.ndarray) -> np.ndarray:
        """Convert a boson basis 2-index matrix to a full composite matrix.

        Args:
            mat (np.ndarray): 2D array of shape (norb, norb).

        Returns:
            np.ndarray: 2D array of shape (n^2, n^2).
        """

        nind = len(self.find)**2
        norb = len(self.bind)

        c2b = np.asarray(self.c2b, dtype=np.int64)
        matret = np.zeros((nind,nind),dtype=np.complex128,order='F')

        # for jorb in range(norb):
        #     for iorb in range(norb):
        #         ind1 = self.c2b[iorb]
        #         ind2 = self.c2b[jorb]
        #         matret[ind1,ind2] = mat[iorb,jorb]

        rhs = np.asarray(matin, dtype=np.complex128).astype(np.complex128)

        matret[np.ix_(c2b, c2b)] = rhs
        return matret ## construct

    def Kpath(self, kpath: list = None, nk: int = None) -> np.ndarray:
        """Generate k-point path through specified high-symmetry points.

        Args:
            kpath (list of list[float]): Sequence of k-point coordinates.
            nk (int): Total number of points along the path.

        Returns:
            None
        """

        kpath = np.array(kpath,dtype=float)
        nnod = kpath.shape[0]
        kmat = np.linalg.inv(np.dot(self.avec,self.avec.T))
        knode = np.zeros(nnod,dtype=float)
        for n in range(1,nnod):
            dk = kpath[n] - kpath[n-1]
            l = np.sqrt(dk@(kmat@dk))
            knode[n] = knode[n-1]+l



        indnod = []
        for n in range(1,nnod-1):
            if n == 1:
                indnod.append(0)
            frac = knode[n]/knode[-1]
            indnod.append(int(round(frac*(nk-1))))
        indnod.append(nk-1)

        kdist = np.zeros(nk,dtype=float)
        kvec = np.zeros((nk,kpath.shape[1]),dtype=float)
        kvec[0] = kpath[0]

        for i in range(1,nnod):
            n1 = indnod[i-1]
            n2 = indnod[i]
            kd1 = knode[i-1]
            kd2 = knode[i]
            k1 = kpath[i-1]
            k2 = kpath[i]
            # print(n1,n2,kd1,kd2,k1,k2)
            for j in range(n1,n2+1):
                frac = float(j-n1)/float(n2-n1)
                kdist[j] = kd1 + frac*(kd2-kd1)
                kvec[j] = k1 + frac*(k2-k1)

        self.kpath = kvec
        self.kdist = kdist
        self.knode = knode

        return None


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
            # probspace[key] = []
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
                            # bind = self.b2f.index([iorb,jorb])
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


    # def indexing(self, ntot, ndivision, divisionarray, flag, n1, n2):
    #     """Map between flat index and multi-dimensional indices.

    #     Args:
    #         ntot (int): Total number of elements.
    #         ndivision (int): Number of dimensions.
    #         divisionarray (list of int): Size of each dimension.
    #         flag (int): Mode flag (1 for encode, 0 for decode).
    #         n1 (int): Input or output flat index.
    #         n2 (list of int): Input or output multi-dimensional index list.

    #     Returns:
    #         tuple: (n1, n2) updated by the indexing operation.
    #     """
    #     tmpsize = 1
    #     for size in divisionarray:
    #         tmpsize *= size

    #     if tmpsize != ntot:
    #         print('array_division wrong')
    #         return

    #     if flag == 1:
    #         n1 = n2[0]
    #         for ii in range(1, ndivision):
    #             tempcnt = 1
    #             for jj in range(ii):
    #                 tempcnt *= divisionarray[jj]
    #             n1 += (n2[ii] ) * tempcnt
    #     else:
    #         n2_array = [0] * ndivision
    #         tempcnt = n1
    #         for ii in range(ndivision - 1):
    #             n2_array[ii] = tempcnt - ((tempcnt) // divisionarray[ii]) * divisionarray[ii]
    #             tempcnt = (tempcnt - n2_array[ii])//divisionarray[ii]
    #         n2_array[ndivision - 1] = tempcnt

    #         # Copy the values from the temporary array to the n2 output array
    #         for i in range(ndivision):
    #             n2[i] = n2_array[i]

    #     return n1, n2

    def FindPositions(self, array, value):
        """Find all positions of a value in a 2D array.

        Args:
            array (iterable of iterable): 2D array to search.
            value: Value to search for.

        Returns:
            list of [int, int]: List of [row_index, col_index] where value matches.
        """
        positions = []
        for row_index, row in enumerate(array):
            for col_index, col_value in enumerate(row):
                if col_value == value:
                    positions.append([row_index, col_index])
        return positions

    def R2mRMapping(self) -> None:
        """Compute mapping indices from k-point grid to its complement (1 - k).

        Returns:
            None
        """
        rkvec = self.kpoint

        mrkvec = np.array(1.0-rkvec,dtype=float)

        for ii in range(mrkvec.shape[0]):
            for jj in range(mrkvec.shape[1]):
                if mrkvec[ii,jj] == 1.0:
                    mrkvec[ii,jj] = 0.0

        mappingidx = []

        for ii in range(rkvec.shape[0]):
            for jj in range(mrkvec.shape[0]):
                if (abs(rkvec[ii,0]-mrkvec[jj,0])<=1.0e-6)and(abs(rkvec[ii,1]-mrkvec[jj,1])<=1.0e-6)and(abs(rkvec[ii,2]-mrkvec[jj,2])<=1.0e-6):
                    mappingidx.append([ii,jj])

        self.mappingidx = mappingidx
        return None
    
    def R2mR(self, matin : np.ndarray) -> np.ndarray:

        self.R2mRMapping()

        matout = np.zeros_like(matin, dtype=np.complex128, order='F')

        for rp in self.mappingidx:
            matout[..., rp[0],:] = matin[..., rp[1], :]

        return matout

    def RT2mRmT(self, G: np.ndarray) -> np.ndarray:
        """Apply the time-reversed mapping to Green's function tensor.

        Args:
            G (np.ndarray): Green's function of shape (norb, norb, ns, nr, ntau).

        Returns:
            np.ndarray: Transformed Green's function with same shape.
        """
        self.R2mRMapping()

        norb = G.shape[0]
        ns = G.shape[2]
        nr = G.shape[3]
        ntau = G.shape[4]

        GmRmT = np.zeros((norb,norb,ns,nr,ntau),dtype=np.complex64,order='F')

        for itau in range(ntau):
            for rp in self.mappingidx:
                for js in range(ns):
                    for iorb in range(norb):
                        for jorb in range(norb):
                            GmRmT[iorb,jorb,js,rp[0],itau] = -G[iorb,jorb,js,rp[1],ntau-itau-1]

        return GmRmT

    def RVec(self, grid : list = None) -> tuple:
        """Generate real-space vector mappings for the k-point grid.

        Returns:
            None
        """
        if (grid == None):
            grid = self.rkgrid
        r = np.zeros((grid[0]*grid[1]*grid[2], 3), dtype=float)
        rind = np.zeros((grid[0]*grid[1]*grid[2],3),dtype=float) 
        from .utility.Common import Common

        for iz in range(grid[2]):
            for iy in range(grid[1]):
                for ix in range(grid[0]):
                    nn1 = [ix,iy,iz]
                    ind1, nn1 = Common.Indexing(grid[0]*grid[1]*grid[2],3,grid,1,0,nn1)
                    if (ix > grid[0]//2):
                        xx = ix-grid[0]
                    else:
                        xx = ix
                    if (iy > grid[1]//2):
                        yy = iy-grid[1]
                    else:
                        yy = iy
                    if (iz > grid[2]//2):
                        zz = iz-grid[2]
                    else:
                        zz = iz
                    r[ind1] = [xx,yy,zz]
                    rind[ind1] = [ix,iy,iz]

        if (grid == self.rkgrid):
            self.rvec = r
            self.rind = rind

        return (r, rind)
    
    def T2mT(self, G : np.ndarray) -> np.ndarray:

        norb = G.shape[0]
        ns = G.shape[2]
        # nr = G.shape[3]
        ntau = G.shape[3]

        tempmat = np.zeros((norb,norb,ns,ntau), dtype=np.complex128, order='F')

        for itau in range(ntau):
            for js in range(ns):
                for jorb in range(norb):
                    for iorb in range(norb):
                        tempmat[iorb,jorb,js,itau] = - G[iorb,jorb,js,ntau-itau-1]

        return tempmat
    
    def K2K3D(self, grid : list = None):

        if grid is None:
            grid = self.rkgrid
        nk = grid[0]*grid[1]*grid[2]
        kind = {}
        for ik in range(nk):
            [n1, n2] = Common.Indexing(nk, 3, grid, 0, ik, [0, 0, 0])
            kind[n1] = n2
        
        if grid == self.rkgrid:
            self.kind = kind

        return kind
    
    def SplitKind(self, kidx : int, kind : dict = None) -> list:
        """Split a k-point index into its 3D components.

        Args:
            kidx (int): Index in the k-point grid.

        Returns:
            list: [kx, ky, kz] corresponding to the 3D k-point coordinates.
        """
        if (kind is None):
            kind = self.kind
        if kidx in kind:
            return kind[kidx]
        else:
            raise ValueError(f"Invalid k-point index: {kidx}")
        
    def MergeKind(self, klist : list, kind : dict = None) -> int:
        """Merge 3D k-point components into a single index.

        Args:
            klist (list): [kx, ky, kz] 3D k-point coordinates.

        Returns:
            int: Merged index in the k-point grid.
        """
        if (kind is None):
            kind = self.kind
        for key, value in kind.items():
            if value == klist:
                return key
            # else:
            #     raise ValueError(f"Invalid k-point components: {klist}")

    def MappingKpoint(self, kpoint : np.ndarray) -> list:

        kpoint_temp = np.zeros_like(kpoint, dtype=np.float64)
        for ik in range(kpoint.shape[0]):
            kx, ky, kz = kpoint[ik]
            if (abs(kx) < 1.0e-6):
                kx_new = 0
            else:
                kx_new = kx % 1
            if (abs(ky) < 1.0e-6):
                ky_new = 0
            else:
                ky_new = ky % 1
            if (abs(kz) < 1.0e-6):
                kz_new = 0
            else:
                kz_new = kz % 1
            kpoint_temp[ik] = [kx_new, ky_new, kz_new]

        mapping = []
        
        for i, pt in enumerate(self.kpoint):
            diff = np.linalg.norm(kpoint_temp - pt, axis = 1)
            idx = np.argmin(diff)
            mapping.append(idx)

        self.mappingkp = mapping

        return None

    def KPoint(self, grid : list) -> np.ndarray:

        kpoint_temp=np.array(list(itertools.product(np.linspace(0,1,num=grid[2],endpoint=False),np.linspace(0,1,num=grid[1],endpoint=False),np.linspace(0,1,num=grid[0],endpoint=False))))
        kpoint=np.fliplr(kpoint_temp)

        return kpoint
    
    def MappingRVec(self, rvec : np.ndarray) -> dict:

        mapping = {}

        for i, pt in enumerate(self.rvec):
            diff = np.linalg.norm(rvec - pt, axis = 1)
            idx = np.argmin(diff)
            mapping[i] = idx

        self.mappingrvec = mapping

        return None

    def MappingBosonFermion(self, iorb):

        [a, [m1, m4]] = self.BAtomOrb(iorb)

        iorbc = self.FIndex([a, m1])
        lorbc = self.FIndex([a, m4])

        return iorbc, lorbc