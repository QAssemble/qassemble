"""Input parsing and execution driver for QAssemble workflows."""
from __future__ import annotations

import ast
import copy
import datetime
import os
from pathlib import Path
import sys
import time
import warnings

import h5py

from .CorrelationFunction import CorrelationFunction


DEFAULT_INPUT_FILE = "qassemble.in"
LEGACY_INPUT_FILE = "input.ini"
REQUIRED_INPUT_SECTIONS = ("Crystal", "Hamiltonian", "Control")


class InputFormatError(ValueError):
    """Raised when a QAssemble input file is not valid structured input."""


def resolve_input_file(input_file: str | Path | None = None) -> Path:
    """Return the input file to read, applying the default/legacy search order."""

    if input_file is not None:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"QAssemble input file not found: {path}")
        return path.resolve()

    default_path = Path(DEFAULT_INPUT_FILE)
    if default_path.exists():
        return default_path.resolve()

    legacy_path = Path(LEGACY_INPUT_FILE)
    if legacy_path.exists():
        return legacy_path.resolve()

    raise FileNotFoundError(
        f"QAssemble input file not found. Expected {DEFAULT_INPUT_FILE}"
        f" or deprecated {LEGACY_INPUT_FILE} in the current directory."
    )


def load_input_file(path: str | Path) -> dict:
    """Load a QAssemble input file without executing arbitrary Python code."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.name == LEGACY_INPUT_FILE:
        warnings.warn(
            f"{LEGACY_INPUT_FILE} is deprecated; use {DEFAULT_INPUT_FILE}.",
            FutureWarning,
            stacklevel=2,
        )
        data = _load_legacy_assignments(text, path)
    else:
        data = _load_top_level_dict(text, path)

    _validate_input_sections(data, path)
    return data


def _load_top_level_dict(text: str, path: Path) -> dict:
    """Parse the current input format: a single top-level dictionary literal."""

    try:
        tree = ast.parse(text, filename=str(path), mode="eval")
        data = ast.literal_eval(tree.body)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise InputFormatError(
            f"{path}: expected a single top-level dictionary literal."
        ) from exc

    if not isinstance(data, dict):
        raise InputFormatError(f"{path}: top-level input must be a dictionary.")
    return data


def _load_legacy_assignments(text: str, path: Path) -> dict:
    """Parse deprecated input.ini files containing only section assignments."""

    try:
        tree = ast.parse(text, filename=str(path), mode="exec")
    except SyntaxError as exc:
        raise InputFormatError(f"{path}: invalid legacy input syntax.") from exc

    data = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            raise InputFormatError(
                f"{path}: legacy input.ini may only contain Crystal,"
                " Hamiltonian, and Control assignments."
            )
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise InputFormatError(
                f"{path}: legacy input assignments must use simple names."
            )

        name = node.targets[0].id
        if name not in REQUIRED_INPUT_SECTIONS:
            raise InputFormatError(
                f"{path}: unsupported legacy assignment {name!r}; expected only"
                " Crystal, Hamiltonian, and Control."
            )
        if name in data:
            raise InputFormatError(f"{path}: duplicate {name} section.")

        try:
            data[name] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise InputFormatError(
                f"{path}: {name} must be a Python literal value."
            ) from exc

    return data


def _validate_input_sections(data: dict, path: Path) -> None:
    """Validate top-level QAssemble input sections."""

    sections = set(data.keys())
    required = set(REQUIRED_INPUT_SECTIONS)
    missing = required - sections
    extra = sections - required

    if missing:
        names = ", ".join(sorted(missing))
        raise InputFormatError(f"{path}: missing required section(s): {names}.")
    if extra:
        names = ", ".join(sorted(str(key) for key in extra))
        raise InputFormatError(f"{path}: unsupported top-level section(s): {names}.")


class Run:
    """Input-file runner for QAssemble command-line calculations."""
    def __init__(self, test=False, input_file=None) -> None:
        """Initialize the runner and optionally execute the configured workflow."""

        self.control = None
        self.func = None
        self.input_file = resolve_input_file(input_file)
        self.ReadInput(self.input_file)
        if test:
            control = self.control
            func = CorrelationFunction(
                cry=control["crystal"],
                ft=control["ft"],
                c=control["run"]["cw"],
            )
            self.func = func
        else:
            if (
                (self.control["run"]["method"] == "tb")
                | (self.control["run"]["method"] == "hf")
                | (self.control["run"]["method"] == "gw")
            ):
                self.RunDiagE()

    def CheckKeyinString(self, key: str, dictionary: dict):
        """Validate that a required key is present in an input dictionary."""

        if key not in dictionary:
            print("missing '" + key + "' in " + dictionary["name"], flush=True)
            sys.exit()
        return None

    def ReadInput(self, input_file=None):
        """Read and parse the QAssemble input file."""

        if input_file is None:
            input_file = self.input_file
        loc = load_input_file(input_file)

        control = {}
        control["name"] = "control"
        control["crystal"] = {}
        control["ft"] = {}
        control["ham"] = {}
        control["run"] = {}
        inicrystal = loc["Crystal"]
        ham = loc["Hamiltonian"]
        ini = loc["Control"]
        ini["name"] = "Control"
        self.CheckKeyinString("Method", ini)
        self.CheckKeyinString("Prefix", ini)
        control["run"]["fn"] = ini.get("Prefix", "glob")
        control["run"]["method"] = ini.get("Method")
        control["run"]["mode"]   = ini.get("Mode", "FromScratch")

        # if control["run"}]
        if os.path.exists(control["run"]["fn"] + ".h5"):
            file = h5py.File(control["run"]["fn"] + ".h5", "r")
            group = file["input"]
            d1 = self.Hdf52Dict(group)
            print(self.CheckInput(d1=d1, d2=loc))
            testloc = self.ChangeInput(copy.deepcopy(loc))
            if self.CheckInput(d1=d1, d2=testloc):
                pass
            else:
                print("Please change the prefix of hdf5 file")
                sys.exit()
        else:
            file = h5py.File(control["run"]["fn"] + ".h5", "w")
            group = file.create_group("input")
            self.Dict2Hdf5(loc, group)
            file.close()
        # file.close()

        inicrystal["name"] = "Crystal"
        ham["name"] = "Hamiltonian"
        ham["OneBody"]["name"] = "OneBody"
        ham["TwoBody"]["name"] = "TwoBody"
        ham["TwoBody"]["Local"]["name"] = "Local"
        ini["name"] = "Control"

        ######## Construct Crystal Structure ########
        self.CheckKeyinString("RVec", inicrystal)
        Rvec = inicrystal["RVec"]
        self.CheckKeyinString("Basis", inicrystal)
        Basis = inicrystal["Basis"]
        CorF = inicrystal.get("CorF", "F")
        Nspin = inicrystal.get("NSpin", 1)
        SOC = inicrystal.get("SOC", False)
        self.CheckKeyinString("KGrid", inicrystal)
        KGrid = inicrystal["KGrid"]
        self.CheckKeyinString("NElec", inicrystal)
        NElec = inicrystal["NElec"]
        control["crystal"]["RVec"] = Rvec
        control["crystal"]["CorF"] = CorF
        control["crystal"]["Basis"] = Basis
        control["crystal"]["NSpin"] = Nspin
        control["crystal"]["SOC"] = SOC
        control["crystal"]["KGrid"] = KGrid
        control["crystal"]["NElec"] = NElec
        # control["crystal"]["orbital"] = orboption

        ######## Construct One-Body Hamiltonian ########
        self.CheckKeyinString("OneBody", ham)
        self.CheckKeyinString("Hopping", ham["OneBody"])
        self.CheckKeyinString("Onsite", ham["OneBody"])
        self.CheckKeyinString("TwoBody", ham)
        self.CheckKeyinString("Local", ham["TwoBody"])
        self.CheckKeyinString("Parameter", ham["TwoBody"]["Local"])

        control["ham"]["hoppinglist"] = ham["OneBody"]["Hopping"]
        control["ham"]["onsitelist"] = ham["OneBody"]["Onsite"]
        control["ham"]["spin"] = ham["OneBody"].get("Spin", False)
        control["ham"]["site"] = ham["OneBody"].get("Site", False)
        control["ham"]["asite"] = ham["OneBody"].get("AntiSite", False)
        control["ham"]["valley"] = ham["OneBody"].get("Valley", False)
        control["ham"]["avalley"] = ham["OneBody"].get("AntiValley", False)
        control["ham"]["aferro"] = ham["OneBody"].get("AntiFerro", False)

        ######## Construct Two-Body Hamiltonian ########
        control["ham"]["coulomb"] = {}
        vlocparameter = {}
        vlocparameter["option"] = {}
        vlocparameter["Parameter"] = ham["TwoBody"]["Local"].get(
            "Parameter", "SlaterKanamori"
        )

        if vlocparameter["Parameter"] == "SlaterKanamori":
            for key, val in ham["TwoBody"]["Local"]["option"].items():
                l = val.get("l", 0)
                U = val.get("U", 0)
                J = val.get("J", 0)
                Up = val.get("Up", U - 2 * J)
                (atom, orb) = key
                if type(orb) == int:
                    orblist = [orb]
                elif type(orb) == tuple:
                    orblist = list(orb)
                vlocparameter["option"][atom + 1] = {}
                vlocparameter["option"][atom + 1]["l"] = l
                vlocparameter["option"][atom + 1]["value"] = [U, Up, J]
                vlocparameter["option"][atom + 1]["orbitals"] = orblist
        if vlocparameter["Parameter"] == "Slater":
            for key, val in ham["TwoBody"]["Local"]["option"].items():
                (atom, orb) = key
                l = val.get("l", 0)
                value = []
                if l == 0:
                    F0 = val.get("F0", 0)
                    value = [F0]
                elif l == 1:
                    F0 = val.get("F0", 0)
                    F2 = val.get("F2", 0)
                    value = [F0, F2]
                elif l == 2:
                    F0 = val.get("F0", 0)
                    F2 = val.get("F2", 0)
                    F4 = val.get("F4", 0)
                    value = [F0, F2, F4]
                elif l == 3:
                    F0 = val.get("F0", 0)
                    F2 = val.get("F2", 0)
                    F4 = val.get("F4", 0)
                    F6 = val.get("F6", 0)
                    value = [F0, F2, F4, F6]
                if type(orb) == int:
                    orblist = [orb]
                elif type(orb) == tuple:
                    orblist = list(orb)
                vlocparameter["option"][atom + 1] = {}
                vlocparameter["option"][atom + 1]["l"] = l
                vlocparameter["option"][atom + 1]["value"] = value
                vlocparameter["option"][atom + 1]["orbitals"] = orblist
        if vlocparameter["Parameter"] == "Kanamori":
            for key, val in ham["TwoBody"]["Local"]["option"].items():
                l = val.get("l", 0)
                U = val.get("U", 0)
                J = val.get("J", 0)
                Up = val.get("Up", U - 2 * J)
                (atom, orb) = key
                if type(orb) == int:
                    orblist = [orb]
                elif type(orb) == tuple:
                    orblist = list(orb)
                vlocparameter["option"][atom + 1] = {}
                vlocparameter["option"][atom + 1]["l"] = l
                vlocparameter["option"][atom + 1]["value"] = [U, Up, J]
                vlocparameter["option"][atom + 1]["orbitals"] = orblist
        # print(vlocparameter)

        control["ham"]["coulomb"]["local"] = vlocparameter

        vnonlocparameter = None
        NonLoc = ham["TwoBody"]["NonLocal"]
        if isinstance(NonLoc, dict):
            ohno = NonLoc.get("Ohno", False)
            jth = NonLoc.get("JTH", False)
            oy = NonLoc.get('OhnoYukawa', False)
        else:
            ohno = False
            jth = False
            oy = False
        # print(jth)
        if ham["TwoBody"]["NonLocal"] == "None":
            control["ham"]["coulomb"]["nonlocal"] = vnonlocparameter
            control["ham"]["coulomb"]["ohno"] = ohno
            control["ham"]["coulomb"]["jth"] = jth
            control["ham"]["coulomb"]["ohnoyuka"] = oy
        elif ohno:
            # vnonlocparameter = OhnoParameterization(U, rkgrid, orboption, lattice, inicrystal["pos"])
            ohno = True
            control["ham"]["coulomb"]["ohno"] = ohno
            control["ham"]["coulomb"]["nonlocal"] = ham["TwoBody"]["NonLocal"]["Vij0"]
            control["ham"]["coulomb"]["jth"] = jth
            control["ham"]["coulomb"]["ohnoyuka"] = oy
        elif jth:
            control["ham"]["coulomb"]["jth"] = jth
            control["ham"]["coulomb"]["ohno"] = ohno
            control["ham"]["coulomb"]["nonlocal"] = vnonlocparameter
            control["ham"]["coulomb"]["ohnoyuka"] = oy
        elif oy:
            control["ham"]["coulomb"]["jth"] = jth
            control["ham"]["coulomb"]["ohno"] = ohno
            control["ham"]["coulomb"]["nonlocal"] = vnonlocparameter
            control["ham"]["coulomb"]["ohnoyuka"] = oy
        else:

            control["ham"]["coulomb"]["nonlocal"] = ham["TwoBody"]["NonLocal"]
            control["ham"]["coulomb"]["ohno"] = ohno
            control["ham"]["coulomb"]["jth"] = jth
            control["ham"]["coulomb"]["ohnoyuka"] = oy


        ######## Check the method ########

        control["run"]["mix"] = ini.get("Mix", 0.1)
        control["run"]["nscf"] = ini.get("NSCF", 100)
        control["run"]["cw"] = ini.get("ConstantW", 1.0)

        # CheckKeyinString("MatsubaraMesh",ini)
        cutoff = ini.get("MatsubaraCutOff", 50)
        kb = 8.6173303 * 10**-5
        if ("T" not in ini) and ("beta" not in ini):
            print("missing T and beta in '" + ini["name"])
            sys.exit()
        if ("T" not in ini) and ("beta" in ini):
            beta = ini.get("beta", 100)
            T = 1 / (beta * kb)
        if ("T" in ini) and ("beta" not in ini):
            T = ini.get("T", 300)
            beta = 1 / (T * kb)

        control["ft"]["T"] = T
        control["ft"]["beta"] = beta
        control["ft"]["cutoff"] = cutoff

        self.control = control

        return None

    def Dict2Hdf5(self, d: dict, h5file: h5py.File):
        """Serialize a nested dictionary into an HDF5 file."""
        for key, value in d.items():
            if isinstance(value, dict):
                group = h5file.create_group(str(key))
                # pprint.pprint(key,value)
                self.Dict2Hdf5(value, group)
            elif isinstance(value, list):
                h5file[str(key)] = str(value)
            else:
                h5file[str(key)] = value

        return None

    def Hdf52Dict(self, h5file: h5py.File):
        """Load a nested dictionary from an HDF5 file."""
        def LoadDict(group: h5py.File):
            """Recursively load an HDF5 group into a dictionary."""
            d = {}
            for key, item in group.items():
                if isinstance(item, h5py.Group):
                    d[key] = LoadDict(item)
                else:
                    if isinstance(item[()], bytes):
                        d[key] = str(item[()], "utf-8")
                    else:
                        d[key] = item[()]
            for key, item in group.attrs.items():
                d[key] = item
            return d

        return LoadDict(h5file)

    def CheckInput(self, d1: dict, d2: dict):
        """
        Compare the input file
        d1 : already saved input file
        d2 : new input file
        """

        for key, value in d2.items():

            if isinstance(value, dict):
                # self.CheckInput()
                self.CheckInput(d1[str(key)], d2[key])
            else:
                val1 = d1[str(key)]
                val2 = d2[key]
                if isinstance(val1, bytes):
                    if str(val1, "utf-8") == str(val2):
                        return True
                    else:
                        print(key, val1, val2)
                        return False
                else:
                    if str(val1) == str(val2):
                        return True
                    else:
                        if key == "Method":
                            continue
                        else:
                            print(key, val1, val2)
                            return False

    def ChangeInput(self, d: dict):
        """Normalize loaded input values for runtime use."""

        dtemp = {}
        for key, val in d.items():
            if isinstance(val, dict):
                valtemp = self.ChangeInput(val)
                dtemp[str(key)] = valtemp
            else:
                if isinstance(val, list):
                    dtemp[str(key)] = str(val)
                else:
                    dtemp[str(key)] = val
        return dtemp

    def CompareDict(self, d1: dict, d2: dict):
        """Compare input dictionaries while tolerating array values."""

        check = []

        for key in d2.keys():
            for key2 in d2[key].keys():
                if key2 == "Method":
                    continue
                elif d1[key][key2] == d2[key][key2]:
                    check.append(1)
                else:
                    check.append(0)
        return check

    def CheckInput(self, d1: dict, d2: dict):
        """Validate that restart input matches the saved input."""

        checklist = self.CompareDict(d1, d2)
        check = True
        if 0 in checklist:
            check = False

        return check

    def RunDiagE(self):
        """Run the diagonalization-only workflow."""

        control = self.control
        cry = control["crystal"]
        ft = control["ft"]
        func = CorrelationFunction(cry=cry, ft=ft)

        itermax = control["run"]["nscf"]
        mix = control["run"]["mix"]
        method = control["run"]["method"]
        fn = control["run"]["fn"]

        if method == "tb":
            print("Tight-Binding calculation start")
            hopping = control["ham"]["hoppinglist"]
            onsite = control["ham"]["onsitelist"]
            spin = control["ham"]["spin"]
            valley = control["ham"]["valley"]
            site = control["ham"]["site"]
            func.TightBinding(
                hopping=hopping, onsite=onsite, spin=spin, valley=valley, site = site, hdf5file=fn + ".h5"
            )
            print("Tight-Binding calculation finish")
        if method == "hf":
            print("Hartree-Fock calculation start")

            hoppinglist = control["ham"]["hoppinglist"]
            onsitelist = control["ham"]["onsitelist"]
            spin = control["ham"]["spin"]
            aferro = control["ham"]["aferro"]
            site = control["ham"]["site"]
            asite = control["ham"]["asite"]
            valley = control["ham"]["valley"]
            avalley = control["ham"]["avalley"]
            vloc = control["ham"]["coulomb"]["local"]
            vnonloc = control["ham"]["coulomb"]["nonlocal"]
            ohno = control["ham"]["coulomb"]["ohno"]
            jth = control["ham"]["coulomb"]["jth"]
            oy = control["ham"]["coulomb"]["ohnoyuka"]
            mode = control["run"]["mode"]
            start = time.time()
            func.HartreeFock(
                itermax=itermax,
                mix=mix,
                mode=mode,
                hopping=hoppinglist,
                onsite=onsitelist,
                spin=spin,
                valley=valley,
                avalley=avalley,
                site = site,
                asite=asite,
                aferro=aferro,
                loccoulomb=vloc,
                nonloccoulomb=vnonloc,
                ohno=ohno,
                jth=jth,
                ohnoyuka = oy,
                hdf5file=fn + ".h5",
            )
            end = time.time()
            print("Hartree-Fock calculation finish")
            delta = datetime.timedelta(seconds=(end - start))
            print(f"Hartree-Fock loop time = {delta}")

        if method == "gw":
            print("GW calculation start")

            hoppinglist = control["ham"]["hoppinglist"]
            onsitelist = control["ham"]["onsitelist"]
            spin = control["ham"]["spin"]
            valley = control["ham"]["valley"]
            site = control["ham"]["site"]
            vloc = control["ham"]["coulomb"]["local"]
            vnonloc = control["ham"]["coulomb"]["nonlocal"]
            ohno = control["ham"]["coulomb"]["ohno"]
            jth = control["ham"]["coulomb"]["jth"]
            oy = control["ham"]["coulomb"]["ohnoyuka"]
            start = time.time()
            func.GWApproximation(
                itermax=itermax,
                mix=mix,
                hoppinglist=hoppinglist,
                onsitelist=onsitelist,
                spin=spin,
                valley=valley,
                site = site,
                loccoulomb=vloc,
                nonloccoulomb=vnonloc,
                ohno=ohno,
                jth=jth,
                ohnoyuka=oy,
                hdf5file=fn + ".h5",
            )
            end = time.time()
            print("GW calculation finish")
            delta = datetime.timedelta(seconds=(end - start))
            print(f"GW loop time = {delta}")

        return None
