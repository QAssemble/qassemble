import copy
import datetime
import logging
import os
import sys
import time

import h5py

from .CorrelationFunction import CorrelationFunction
from .Crystal import Crystal
from .utility.DLR import DLR

logger = logging.getLogger("QAssemble")


class Run:
    def __init__(self, test=False) -> None:

        self.control = None
        self.func = None
        self.ReadInput()
        if test:
            control = self.control
            func = self.BuildCorrelationFunction(control)
            self.func = func
        else:
            if (
                (self.control["run"]["method"] == "tb")
                | (self.control["run"]["method"] == "hf")
                | (self.control["run"]["method"] == "gw")
            ):
                self.RunDiagE()

    def CheckKeyinString(self, key: str, dictionary: dict):

        if key not in dictionary:
            logger.error("missing '" + key + "' in " + dictionary["name"])
            sys.exit()
        return None

    def BuildCorrelationFunction(self, control: dict):

        try:
            return CorrelationFunction(control=control)
        except Exception:
            func = CorrelationFunction.__new__(CorrelationFunction)
            func.control = control
            func.c = control["run"]["cw"]
            func.niham = None
            func.green = None
            func.greenbare = None
            func.sigmah = None
            func.sigmaf = None
            func.sigmagwc = None
            func.ham = None
            func.occ = None
            func.vbare = None
            func.pol = None
            func.w = None
            func.crystal = Crystal(cry=control["crystal"])
            func.dlr = DLR(control["ft"])

            return func

    def ReadInput(self):

        loc = {}
        glob = {}
        exec(open("input.ini").read(), glob, loc)

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
            logger.info(self.CheckInput(d1=d1, d2=loc))
            testloc = self.ChangeInput(copy.deepcopy(loc))
            if self.CheckInput(d1=d1, d2=testloc):
                pass
            else:
                logger.error("Please change the prefix of hdf5 file")
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

        control["ham"]["onebody"] = copy.deepcopy(ham["OneBody"])

        ######## Construct Two-Body Hamiltonian ########
        control["ham"]["twobody"] = copy.deepcopy(ham["TwoBody"])


        ######## Check the method ########

        control["run"]["mix"] = ini.get("Mix", 0.1)
        control["run"]["nscf"] = ini.get("NSCF", 100)
        control["run"]["cw"] = ini.get("ConstantW", 1.0)

        # CheckKeyinString("MatsubaraMesh",ini)
        cutoff = ini.get("MatsubaraCutOff", 50)
        kb = 8.6173303 * 10**-5
        if ("T" not in ini) and ("beta" not in ini):
            logger.error("missing T and beta in '" + ini["name"])
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
        def LoadDict(group: h5py.File):
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
                        logger.warning(f"{key} {val1} {val2}")
                        return False
                else:
                    if str(val1) == str(val2):
                        return True
                    else:
                        if key == "Method":
                            continue
                        else:
                            logger.warning(f"{key} {val1} {val2}")
                            return False

    def ChangeInput(self, d: dict):

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

        checklist = self.CompareDict(d1, d2)
        check = True
        if 0 in checklist:
            check = False

        return check

    def RunDiagE(self):

        control = self.control
        func = self.BuildCorrelationFunction(control)

        method = control["run"]["method"]

        if method == "tb":
            logger.info("Tight-Binding calculation start")
            func.TightBinding()
            logger.info("Tight-Binding calculation finish")
        if method == "hf":
            logger.info("Hartree-Fock calculation start")


            start = time.time()
            func.HartreeFock()
            end = time.time()
            logger.info("Hartree-Fock calculation finish")
            delta = datetime.timedelta(seconds=(end - start))
            logger.info(f"Hartree-Fock loop time = {delta}")

        if method == "gw":
            logger.info("GW calculation start")


            start = time.time()
            func.GWApproximation()
            end = time.time()
            logger.info("GW calculation finish")
            delta = datetime.timedelta(seconds=(end - start))
            logger.info(f"GW loop time = {delta}")

        return None
