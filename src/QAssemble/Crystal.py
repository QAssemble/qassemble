import numpy as np
import sys
import itertools
import logging

import copy
from .utility.Common import Common
from .BasisIndex import BasisIndex

logger = logging.getLogger("QAssemble")

class Crystal(object):
    """Handles lattice geometry, orbitals, and basis indexing for quantum assembly calculations.

    This class constructs indices and vectors for fermionic and bosonic orbitals,
    k-point grids, real-space vectors, and provides methods to map between different basis representations.
    Basis indexing logic is delegated to BasisIndex.
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
            Optional keys:
                Species (list of str): Element symbols per basis atom (e.g. ['C', 'C']).
                    If omitted, dummy elements are assigned based on orbital count
                    for symmetry analysis.
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
        self.species = cry.get('Species', None)
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

        svec = np.zeros((3,3),dtype=np.float64)
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

        self.mappingkp = []
        self.orboption = orboption

        self._basis_index = BasisIndex(orboption=orboption, ns=self.ns)
        self.siteorbitals = self.SiteOrbitalDict()

        self.RVec()

        # Symmetry analysis (lazy — computed on first access)
        self._structure = None
        self._symmetry_info = None

        self.PrintSymmetryInfo()

        return None

    # ── BasisIndex delegation: properties ──

    @property
    def find(self):
        return self._basis_index.find

    @property
    def bind(self):
        return self._basis_index.bind

    @property
    def full(self):
        return self._basis_index.full

    @property
    def bbasis(self):
        return self._basis_index.bbasis

    @property
    def pbasis(self):
        return self._basis_index.pbasis

    @property
    def c2b(self):
        return self._basis_index.c2b

    @property
    def forb2atom(self):
        return self._basis_index.forb2atom

    @property
    def borb2atom(self):
        return self._basis_index.borb2atom

    @property
    def probspace(self):
        return self._basis_index.probspace

    @property
    def probindex(self):
        return self._basis_index.probindex

    @property
    def fimpdict(self):
        return self._basis_index.fimpdict

    @property
    def bimpdict(self):
        return self._basis_index.bimpdict

    @property
    def fprojector(self):
        return self._basis_index.fprojector

    @property
    def bprojector(self):
        return self._basis_index.bprojector

    @property
    def fprojector_prob(self):
        return self._basis_index.fprojector_prob

    @property
    def bprojector_prob(self):
        return self._basis_index.bprojector_prob

    # ── BasisIndex delegation: methods ──

    def FAtomOrb(self, key):
        return self._basis_index.FAtomOrb(key)

    def FIndex(self, val):
        return self._basis_index.FIndex(val)

    def BAtomOrb(self, key):
        return self._basis_index.BAtomOrb(key)

    def BIndex(self, val):
        return self._basis_index.BIndex(val)

    def MappingBosonFermion(self, iorb):
        return self._basis_index.MappingBosonFermion(iorb)

    def Composite2OrbSpin(self, mat):
        return self._basis_index.Composite2OrbSpin(mat)

    def OrbSpin2Composite(self, mat):
        return self._basis_index.OrbSpin2Composite(mat)

    def Quad2Double(self, matin):
        return self._basis_index.Quad2Double(matin)

    def Double2Quad(self, matin):
        return self._basis_index.Double2Quad(matin)

    def Full2Quad(self, matin):
        return self._basis_index.Full2Quad(matin)

    def Quad2Full(self, mat):
        return self._basis_index.Quad2Full(mat)

    def Full2Double(self, mat):
        return self._basis_index.Full2Double(mat)

    def Double2Full(self, mat):
        return self._basis_index.Double2Full(mat)


    # ── Crystal-own methods ──

    def SiteOrbitalDict(self, atomsite = None, local : bool = False) -> dict:
        """Build a dictionary of orbital numbers for selected atom sites.

        Args:
            atomsite (None | int | list | tuple | set | dict):
                Site specification.
                - None: all atom sites.
                - int: single atom site index.
                - iterable: collection of atom site indices.
                - dict: use dictionary keys as atom site indices.
            local (bool):
                If True, returns local orbital labels per site (0..n_orb-1).
                If False (default), returns global fermionic orbital indices.

        Returns:
            dict: {atom_site: [orbital_numbers, ...]}.
        """
        if atomsite is None:
            atomsite = list(self.orboption.keys())
        elif isinstance(atomsite, (int, np.integer)):
            atomsite = [int(atomsite)]
        elif isinstance(atomsite, dict):
            atomsite = list(atomsite.keys())
        else:
            atomsite = list(atomsite)

        siteorbitals = {}
        for site in atomsite:
            if not isinstance(site, (int, np.integer)):
                raise TypeError(f"Atom site index must be int, got {type(site)}")

            site = int(site)
            if site not in self.orboption:
                raise ValueError(f"Invalid atom site index: {site}")

            n_orb = self.orboption[site]
            if local:
                siteorbitals[site] = list(range(n_orb))
            else:
                siteorbitals[site] = [self.FIndex([site, m]) for m in range(n_orb)]

        return siteorbitals

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

    # ── Symmetry analysis via pymatgen ──

    def _get_species(self) -> list:
        """Get species list for pymatgen Structure.

        If Species was provided in the input dict, use it directly.
        Otherwise, assign 'X' (dummy species) for all atoms.

        Returns:
            list of str: Element symbols for each basis atom.
        """
        if self.species is not None:
            return list(self.species)

        return ['X'] * len(self.basisf)

    def GetStructure(self):
        """Create and cache a pymatgen Structure from the crystal data.

        Returns:
            pymatgen.core.Structure or None: The crystal structure object,
                or None if pymatgen is not available.
        """
        if self._structure is not None:
            return self._structure

        try:
            from pymatgen.core import Lattice, Structure
        except (ImportError, Exception) as e:
            logger.warning(f"pymatgen is not available ({e}). "
                           "Install/fix pymatgen for symmetry analysis.")
            return None

        lattice = Lattice(self.avec)
        species = self._get_species()
        self._structure = Structure(lattice, species, self.basisf)
        return self._structure

    def GetSymmetryInfo(self, symprec: float = 0.01) -> dict:
        """Perform symmetry analysis and return a summary dictionary.

        Args:
            symprec (float): Symmetry precision tolerance for SpacegroupAnalyzer.

        Returns:
            dict: Symmetry information with keys:
                spacegroup_symbol (str): International spacegroup symbol.
                spacegroup_number (int): International Tables number.
                pointgroup (str): Point group symbol.
                crystal_system (str): e.g. 'hexagonal', 'cubic', etc.
                nsymops (int): Number of symmetry operations.
                wyckoff_symbols (list of str): Wyckoff letters per atom.
                equivalent_atoms (list of int): Equivalent atom mapping.
                lattice_type (str): e.g. 'hexagonal', 'rhombohedral', etc.
                species_used (list of str): Element labels used (real or dummy).
                dummy_species (bool): Whether dummy elements were used.
            Returns None if pymatgen is not available.
        """
        if self._symmetry_info is not None:
            return self._symmetry_info

        structure = self.GetStructure()
        if structure is None:
            return None

        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except (ImportError, Exception) as e:
            logger.warning(f"pymatgen.symmetry is not available ({e}).")
            return None

        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)

        symm_dataset = analyzer.get_symmetry_dataset()

        # Use attribute access (SpglibDataset); fall back to dict access
        try:
            wyckoffs = symm_dataset.wyckoffs
        except AttributeError:
            wyckoffs = symm_dataset.get('wyckoffs', [])

        try:
            equiv = symm_dataset.equivalent_atoms
            equiv = equiv.tolist() if hasattr(equiv, 'tolist') else list(equiv)
        except AttributeError:
            equiv = symm_dataset.get('equivalent_atoms', [])
            if hasattr(equiv, 'tolist'):
                equiv = equiv.tolist()

        self._symmetry_info = {
            'spacegroup_symbol': analyzer.get_space_group_symbol(),
            'spacegroup_number': analyzer.get_space_group_number(),
            'pointgroup': analyzer.get_point_group_symbol(),
            'crystal_system': analyzer.get_crystal_system(),
            'nsymops': len(analyzer.get_symmetry_operations()),
            'wyckoff_symbols': list(wyckoffs),
            'equivalent_atoms': list(equiv),
            'lattice_type': analyzer.get_lattice_type(),
            'species_used': self._get_species(),
            'dummy_species': self.species is None,
        }
        return self._symmetry_info

    def PrintSymmetryInfo(self, symprec: float = 0.01) -> None:
        """Print a formatted summary of the crystal symmetry analysis.

        Args:
            symprec (float): Symmetry precision tolerance.
        """
        info = self.GetSymmetryInfo(symprec=symprec)

        if info is None:
            logger.info("Symmetry analysis unavailable (pymatgen not installed).")
            return

        logger.info("=" * 55)
        logger.info("  Crystal Symmetry Analysis")
        logger.info("=" * 55)
        logger.info(f"  Space group : {info['spacegroup_symbol']} (No. {info['spacegroup_number']})")
        logger.info(f"  Point group : {info['pointgroup']}")
        logger.info(f"  Crystal sys : {info['crystal_system']}")
        logger.info(f"  Lattice type: {info['lattice_type']}")
        logger.info(f"  # Sym. ops  : {info['nsymops']}")
        logger.info("-" * 55)
        logger.info(f"  {'Atom':>4}  {'Species':>8}  {'Wyckoff':>8}  {'Equiv':>5}  {'Orbitals':>8}")
        logger.info("-" * 55)
        for i in range(len(info['species_used'])):
            logger.info(f"  {i:>4}  {info['species_used'][i]:>8}  "
                        f"{info['wyckoff_symbols'][i]:>8}  "
                        f"{info['equivalent_atoms'][i]:>5}  "
                        f"{self.orboption[i]:>8}")
        logger.info("-" * 55)
        if info['dummy_species']:
            logger.info("  Note: Dummy species 'X' used (no Species provided).")
        logger.info("=" * 55)
