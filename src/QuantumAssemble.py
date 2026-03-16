#!/usr/bin/env python3.9
import copy
import datetime
import gc
import importlib
import json
import os
import string
import subprocess
import sys
import time
import h5py

# def ensure_module(module_name, package_name=None):
#     package = package_name or module_name
#     try:
#         return importlib.import_module(module_name)
#     except ModuleNotFoundError as exc:
#         if getattr(exc, "name", module_name) != module_name:
#             raise
#         print(f"Installing missing Python package '{package}' with pip...", flush=True)
#     try:
#         subprocess.check_call(
#             [sys.executable, "-m", "pip", "install", package],
#         )
#     except subprocess.CalledProcessError as exc:
#         print(
#             f"Failed to install required package '{package}': {exc}",
#             flush=True,
#         )
#         raise
#     return importlib.import_module(module_name)


# h5py = ensure_module("h5py")
# np = ensure_module("numpy")
# ensure_module("scipy")
# ensure_module("matplotlib")
# ensure_module("numba")
# ensure_module("finufft")
# ensure_module("mpi4py")
# ensure_module("mpi4py_fft", "mpi4py-fft")

from QAssemble.CorrelationFunction import CorrelationFunction


class Run:
    def __init__(self, test=False) -> None:

        self.control = None
        self.func = None
        self.ReadInput()
        self.mpimanager = None
        if test:
            control = self.control
            func = CorrelationFunction(
                latt=control["crystal"]["lattice"],
                basisposition=control["crystal"]["basispos"],
                ns=control["crystal"]["ns"],
                soc=control["crystal"]["soc"],
                rkgrid=control["crystal"]["rkgrid"],
                orboption=control["crystal"]["orbital"],
                N=control["crystal"]["nume"],
                T=control["ft"]["T"],
                beta=control["ft"]["beta"],
                size=control["ft"]["size"],
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

        if key not in dictionary:
            print("missing '" + key + "' in " + dictionary["name"], flush=True)
            sys.exit()
        return None

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
        ohno = NonLoc.get("Ohno", False)
        jth = NonLoc.get("JTH", False)
        oy = NonLoc.get('OhnoYukawa', False)
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

        # if os.path.exists(control['run']['fn']+'.h5'):
        #     pass
        # else:
        #     with h5py.File(control['run']['fn']+'.h5','w') as file:
        #         group = file.create_group('input')
        #         for key, val in control.items():
        #             group.create_dataset(key,data=val)

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
        func = CorrelationFunction(control=control, mpimanager=self.mpimanager)
        # func = CorrelationFunction(latt=control['crystal']['lattice'], basisposition=control['crystal']['basispos'], ns=control['crystal']['ns'],soc=control['crystal']['soc'],rkgrid=control['crystal']['rkgrid'],orboption=control['crystal']['orbital'],N=control['crystal']['nume'],T=control['ft']['T'],beta=control['ft']['beta'],size=control['ft']['size'],c=control['run']['cw'])

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
            # func.TightBinding(hoppinglist=hoppinglist,onsitelist=onsitelist)
            func.TightBinding(
                hopping=hopping, onsite=onsite, spin=spin, valley=valley, site = site, hdf5file=fn + ".h5"
            )
            print("Tight-Binding calculation finish")
            # flatstc = FLatStc(crystal=func.cry)
            # energy = flatstc.Diagonalize(hamtb)
            # FLatStcSave(hamtb,'hamtb')
            # FLatStcSave(energy,'energy')
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
            # func.HartreeFock(itermax=itermax,mix=mix,hoppinglist=hoppinglist,onsitelist=onsitelist,loccoulomb=vloc,nonloccoulomb=vnonloc,ohno=ohno)
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

            # FLatStcSave(hamhf,'hamhf')
            # FLatStcSave(sigmah.hk,'sigmahk')
            # FLatStcSave(sigmaf.fk,'sigmafk')
            # BLatStcSave(func.vbare.k,'vk')
        # if (method=="gw"):
        #     print("Hartree-Fock calculation start")

        #     hoppinglist = control['ham']['hoppinglist']
        #     onsitelist = control['ham']['onsitelist']
        #     vloc = control['ham']['coulomb']['local']
        #     vnonloc = control['ham']['coulomb']['nonlocal']
        #     ohno = control['ham']['coulomb']['ohno']
        #     start = time.time()
        #     func.HartreeFock(itermax=itermax,mix=mix,hoppinglist=hoppinglist,onsitelist=onsitelist,loccoulomb=vloc,nonloccoulomb=vnonloc,ohno=ohno)
        #     end = time.time()
        #     print("Hartree-Fock calculation finish")
        #     delta = datetime.timedelta(seconds=(end-start))
        #     print(f"Hartree-Fock loop time = {delta}")

        # FLatDynSave(func.green.gkf,'gkfhf')
        # FLatStcSave(func.hamhf,'hamhf')
        # FLatStcSave(func.sigmah.hk,'sigmahk')
        # FLatStcSave(func.sigmaf.fk,'sigmafk')
        # BLatStcSave(func.vbare.k,'vk')
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

            # FLatDynSave(func.green.gkf,'gkf')
            # FLatStcSave(func.sigmah.hk,'sigmahk')
            # FLatStcSave(func.sigmaf.fk,'sigmafk')
            # FLatDynSave(func.sigmac.kf,'sigmackf')
            # BLatDynSave(func.w.wkf,'wkf')
            # BLatDynSave(func.pol.polkf,'pkf')
            # BLatStcSave(func.vbare.k,'vk')
        return None

    # def OhnoParameter(self):
    #     '''
    #     Set the non-loc bare coulomb interaction by using Ohno parameterization

    #     V = U/{\kappa_ij(1+cR_{ij}^2)}^{1/2}
    #     '''
    #     kappa = 2.0
    #     vlist = []
    #     rkgrid = self.control["crystal"]['rkgrid']


if __name__ == "__main__":
    print("Calculation Start")
    run = Run()
    print("Calculation Finish")
