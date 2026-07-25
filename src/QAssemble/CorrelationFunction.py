import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
import gc
import h5py
from .Crystal import Crystal
# from .FTGrid import FTGrid
from .utility.DLR import DLR
from .utility.Convergence import Convergence
from .FLatDyn import *
from .FLatStc import *
from .FLocDyn import *
from .FLocStc import *
from .BLatDyn import *
from .BLatStc import *
from .BLocDyn import *
from .BLocStc import *
from .Projector import Projector
from .Method import HF, HFLoc, GW, GWLoc, ImpurityAction

logger = logging.getLogger("QAssemble")

class CorrelationFunction(object):

    def __init__(self, control : dict):

        self.control = control
        self.c = control["run"]["cw"]

        self.green = None
        self.niham = None
        self.greenbare = None
        self.sigmah = None
        self.sigmaf = None
        self.sigmagwc = None
        self.sigh = None
        self.sigf = None
        self.siggwc = None
        self.ham = None
        self.occ = None
        self.vbare = None
        self.pol = None
        self.w = None
        cry = control['crystal']
        ft = control['ft']
        self.crystal = Crystal(cry=cry)
        self.dlr = DLR(ft)

        
        onebody = control["ham"].get("onebody")

        self.niham = H0(crystal=self.crystal, onebody=onebody, hdf5file=control["run"]["fn"]+'.h5', group='init')

        self.vbare = V(crystal=self.crystal, twobody=control['ham'].get('twobody'), hdf5file=control["run"]["fn"]+'.h5', group='init')

        self.greenbare = G0(crystal=self.crystal, dlr=self.dlr, hamtb=self.niham.k, hdf5file=control["run"]["fn"]+'.h5', group='init')

        # Inject a method-appropriate default HDF5 group for the
        # Convergence mirror BEFORE constructing Convergence (Convergence
        # caches the value in _init_common). User-supplied
        # control['run']['convergence_hdf5_group'] always wins via setdefault.
        _HDF5_GROUP_BY_METHOD = {
            "hf":       "hf/convergence",
            "gw":       "gw/convergence",
            "dmft":     "impurity_solver/convergence",
            "edmft":    "impurity_solver/convergence",
            "gw+edmft": "impurity_solver/convergence",
        }
        _method = control["run"].get("method")
        if _method == "hf" and control["run"].get("mode") == "Restart":
            _default_group = "hf_restart/convergence"
        else:
            _default_group = _HDF5_GROUP_BY_METHOD.get(_method)
        if _default_group is not None:
            control["run"].setdefault("convergence_hdf5_group", _default_group)

        self.conv = Convergence(control)


    def TightBinding(self):

        # file = h5py.File(fn+'.h5','w')
        # tb = file.create_group('tb')

        group = 'tb'
        errmessage = "missing input for tight binding calculation"
        hdf5file = self.control["run"]["fn"] + '.h5'
        # niham = NIHamiltonian(crystal=self.cry,hoppinglist=hoppinglist,onsitelist=onsitelist,hdf5file=tb)
        niham = self.niham
        # file.close()

        return niham

    def HartreeFock(self):

        errmessage = "missing input for HF calculation"

        itermax = self.control["run"]["nscf"]
        mix = self.control["run"]["mix"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        mode = self.control["run"]["mode"]
        group = 'hf'
        mixing_method = self.control["run"]["mixing_method"]
        npulay = int(self.control["run"]["npulay"])

        if (mode == 'FromScratch'):
            
            niham = self.niham
            vbare = self.vbare
            

        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = self.niham
            vbare = self.vbare

        onebody = self.control["ham"].get("onebody")
        # twobody = self.control["ham"].get("twobody")

        self.conv.Start()
        diag_prev_by_key = None

        for iter in range(1, itermax+1):
            if iter==1:
                # onsite_temp = None
                # if self.crystal.ns == 2:
                #     onsite_temp = {}
                #     for js in range(self.crystal.ns):
                #         onsite_temp[js] = {}
                #         for iorb in range(len(self.crystal.find)):
                #             ii, m1 = self.crystal.FAtomOrb(iorb)
                #             if js == 0:
                #                 onsite_temp[js][(ii,m1)] = 1.0
                #             else:
                #                 onsite_temp[js][(ii,m1)] = -1.0 
                if mode == "FromScratch":
                    niham_temp = H0(self.crystal,onebody=onebody, hdf5file=hdf5file,group='test_hf')
                    hold = H(crystal=self.crystal,ham=niham_temp.k,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                elif mode == "Restart":
                    niham_temp = H0(self.crystal,onebody=onebody, hdf5file=None,group='test_hf')
                    glob = h5py.File(hdf5file,'r')
                    hf = glob['hf']
                    hk = hf['Hamiltonian']['hk'][:]
                    glob.close()
                    hold = H(crystal=self.crystal,ham=hk,beta=self.dlr.beta,hdf5file=hdf5file,group=group)
                    
                    
                self.conv.seed_prev("F", hold.k, kind="array")
                self.conv.seed_prev("mu", float(hold.mu), kind="scalar")

            logger.debug(hold.occ)
            sigmah = SigH(crystal=self.crystal,occ=hold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group,iteration=iter)
            sigmaf = SigF(crystal=self.crystal,occr=hold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group,iteration=iter)
            sigmah.Mixing(mix=mix, method=mixing_method, npulay=npulay)
            sigmaf.Mixing(mix=mix, method=mixing_method, npulay=npulay)
            if (iter % 50 == 0):
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
            hnew = H(crystal=self.crystal,ham=niham.k,beta=self.dlr.beta,sigmah=sigmah.k,sigmaf=sigmaf.k,hdf5file=hdf5file,group=group,iteration=iter)
            # hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=None,sigmaf=sigmaf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save('hk')

            self.conv.StartIter(iter)
            self.conv.CheckSelf("F", value=hnew.k, kind="array")
            self.conv.CheckSelf("mu", value=float(hnew.mu), kind="scalar")
            self.conv.RecordDiagnostics({})
            converged, info = self.conv.Commit(iter, will_continue=(iter < itermax))
            fcheck = info["self"]["F"]["abs"]
            logger.info(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if converged:
                logger.info(f"Self-consistency is achived with {iter}-th")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk', scf=False)
                sigmah.Save('sigh', scf=False)
                sigmaf.Save('sigf', scf=False)
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
                break
            elif(iter==itermax):
                logger.warning(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk', scf=False)
                sigmah.Save('sigh', scf=False)
                sigmaf.Save('sigf', scf=False)
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
            else:
                # hnew.OccMixing(iter=iter, mix=mix, occkb = hnew.occk, occkm=hold.occk)
                hold=hnew
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()


    def GWApproximation(self):
        itermax = self.control["run"]["nscf"]
        mix = self.control["run"]["mix"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'gw'
        mixing_method = self.control["run"]["mixing_method"]
        npulay = int(self.control["run"]["npulay"])

        gbare = self.greenbare
        vbare = self.vbare

        self.conv.Start()

        g = G(crystal=self.crystal, dlr=self.dlr, greenbare=gbare.kf, hdf5file=hdf5file, group=group)
        self.conv.seed_prev("F", g.kf, kind="array")
        self.conv.seed_prev("mu", float(g.mu), kind="scalar")
        g.Save('gkf_ini', False)

        pol_zero = np.zeros(vbare.k.shape + (len(self.dlr.nu),), dtype=np.complex128, order="F")
        w = W(crystal=self.crystal, dlr=self.dlr, pol=pol_zero, vbare=vbare, c=self.c, hdf5file=hdf5file, group=group)
        self.conv.seed_prev("B", np.zeros_like(w.kf), kind="array")
        w.Save('wkf_ini', False)
            
        

        for iter in range(1, itermax + 1):
            
            
            hf_method = HF(
                occ=g.occ,
                occr=g.occr,
                v=vbare,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
                mix=mix,
                mixing_method=mixing_method,
                npulay=npulay,
            )
            sigh, sigf = hf_method()

            gw_method = GW(
                g=g,
                w=w,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
                mix=mix,
                mixing_method=mixing_method,
                npulay=npulay,
            )
            siggwc, pol = gw_method()
            pol.Save('pkf')

            wnew = W(crystal=self.crystal, dlr=self.dlr, pol=pol.kf, vbare=vbare, c=self.c, hdf5file=hdf5file, group=group, iteration=iter)
            wnew.Save('wkf')

            siggwc.Save('siggwckf')

            gnew = G(self.crystal, self.dlr, gbare.kf, sigh.k, sigf.k, siggwc.kf, hdf5file=hdf5file, group=group, iteration=iter)
            gnew.Save('gkf')

            self.conv.StartIter(iter, ready_after=npulay)
            self.conv.CheckSelf("F", value=gnew.kf, kind="array")
            self.conv.CheckSelf("B", value=wnew.kf, kind="array")
            self.conv.CheckSelf("mu", value=float(gnew.mu), kind="scalar")
            self.conv.RecordDiagnostics({})
            converged, _ = self.conv.Commit(iter, will_continue=(iter < itermax))

            if converged or iter == itermax:
                if converged:
                    logger.info(f"Self-consistency is achieved at iter {iter}")
                else:
                    logger.warning(f"GW reached max iter {itermax} without convergence")
                self.green = gnew
                self.pol = pol
                self.w = wnew
                self.siggwc = siggwc
                self.sigf = sigf
                self.sigh = sigh
                gnew.Save('gkf', scf=False)
                sigh.Save('sigh', scf=False)
                sigf.Save('sigf', scf=False)
                siggwc.Save('siggwckf', scf=False)
                pol.Save('pkf', scf=False)
                wnew.Save('wkf', scf=False)
                break
            else:
                g = gnew
                w = wnew

                del gnew, sigh, sigf, siggwc, pol, wnew
                gc.collect()

    def DMFT(self):

        errmessage = "missing input for DMFT calculation"

        itermax = self.control["run"]["nscf"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'dmft'
        # self.control["run"]["method"]

        config = self.control.get("impurity", self.control.get("dmft", {}))
        impdict = config.get("impdict", config.get("ImpDict"))
        equiv = config.get("equiv", config.get("Equiv"))

        if impdict is None:
            raise KeyError("DMFT requires control['impurity']['impdict']")
        if equiv is None:
            raise KeyError("DMFT requires control['impurity']['equiv']")

        projector = Projector(basisindex=self.crystal._basis_index,impdict=copy.deepcopy(impdict),equiv=copy.deepcopy(equiv),)
        problem_keys = list(projector.fprojector.keys())
        self.vbare.vloc.projector=projector
        
        sigmah_current = None
        sigmaf_current = None
        sigc_current = None

        green = G(
            crystal=self.crystal,
            dlr=self.dlr,
            greenbare=self.greenbare.kf,
            hdf5file=hdf5file,
            group=group,
        )
        green.Save('gkf_ini', scf=False)

        mix = self.control["run"]["mix"]
        min_iter = self.conv.min_iter

        logger.info(f"[DMFT] mix={mix}, min_iter={min_iter}")

        if itermax <= min_iter:
            logger.warning(f"nscf={itermax} <= min_iter={min_iter}; delta convergence "
                           f"cannot trigger break. Loop will run to max_iter.")

        self.conv.Start()

        for iter in range(1, itermax+1):
            sigc = SigC(crystal=self.crystal, dlr=self.dlr)

            gcheck = 0.0
            
            diag_by_key = {}        # captured immediately inside the per-key loop

            for key in problem_keys:
                gloc = GLoc(crystal=self.crystal,dlr=self.dlr,projector=projector,green=green.kf,key=key,hdf5file=hdf5file,group=group,iteration=iter-1,scf=True)

                eimp = EImp(crystal=self.crystal,projector=projector,key=key,hamtb=self.niham.k,mu=green.mu,hdf5file=hdf5file, group=group, iteration=iter)
                eimp.Save('eimp')

                sighloc = None
                sigfloc = None
                sigcloc = None
                if sigmah_current is not None:
                    sighloc = eimp.Projection(sigmah_current, key)
                if sigmaf_current is not None:
                    sigfloc = eimp.Projection(sigmaf_current, key)
                if sigc_current is not None:
                    sigcloc = gloc.Projection(sigc_current, key)
                print(f"[Hyb-build] key={key}, sighloc[0,0,0]={None if sighloc is None else sighloc[0,0,0]}")

                hyb = Hyb(crystal=self.crystal,dlr=self.dlr,projector=projector,key=key,green=gloc.f,eimp=eimp.e,sigh=sighloc,sigf=sigfloc,sigc=sigcloc,hdf5file=hdf5file,group=group,iteration=iter)
                hyb.Save('hyb')

                fweiss = FWeiss(
                    crystal=self.crystal, dlr=self.dlr, projector=projector,
                    key=key, eimp=eimp, hyb=hyb, mu=green.mu,
                    hdf5file=hdf5file, group=group,
                )

                bweiss = BWeiss(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    vloc=self.vbare.vloc,
                    w=None,
                    p=None,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                bweiss.Save('bweiss')

                impurity_method = ImpurityAction(
                    fweiss=fweiss,
                    bweiss=bweiss,
                    key=key,
                    control=self.control["run"],
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                impurity_result = impurity_method()

                # ---- CTQMC output guard (F12): explicit failure if PostProcessing
                #      produced no sigma/Green's function for this key. ----
                for _attr in ("gimp", "sighimp", "sigfimp", "sigimp"):
                    if getattr(impurity_result, _attr, None) is None:
                        raise RuntimeError(
                            f"CTQMC produced no {_attr} for key={key}, iter={iter}. "
                            f"Inspect params.obs.json or solver stderr."
                        )

                sighimp = impurity_result.sighimp
                sigfimp = impurity_result.sigfimp
                sigimp = impurity_result.sigimp

                sigc.ImpEmbedding(
                    sigimp=sigimp.f,
                    sighimp=sighimp.h,
                    sigfimp=sigfimp.s,
                    projector=projector,
                    key=key,
                )
                diag_by_key[key] = dict(impurity_result.diagnostics)

            sigc()

            green_next = G(
                crystal=self.crystal,
                dlr=self.dlr,
                greenbare=self.greenbare.kf,
                sigmagwc=sigc.kf,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
            )

            # ---- mu NaN guard (F10): catch corrupted chemical potential
            #      before it propagates into the next iter's G. ----
            if not np.isfinite(green_next.mu):
                raise RuntimeError(
                    f"iter {iter}: mu solver produced non-finite mu={green_next.mu}. "
                    f"Likely cause: Sigma corruption (NaN/Inf) or bisection failure."
                )

            green_next.Save('gkf')

            converged = False
            gcheck = float("nan")
            dG_iter = float("nan")
            dmu = float("nan")
            if iter > 1:
                completed_iter = iter - 1
                self.conv.StartIter(completed_iter)
                self.conv.CheckSelfHDF5(
                    name="GLoc",
                    group=group,
                    subgroup="GLoc",
                    current=f"gloc.{completed_iter}",
                    previous=f"gloc.{completed_iter - 1}",
                    keys=problem_keys,
                )
                self.conv.CheckSelf('mu', value=float(green.mu), kind='scalar')
                self.conv.CheckCrossHDF5(
                    name_a = "GLoc",
                    name_b="GImp",
                    group=group,
                    subgroup_a="GLoc",
                    subgroup_b="GImp",
                    stem_a=f"gloc.{completed_iter}",
                    stem_b=f"gimp.{completed_iter}",
                    keys=problem_keys,
                )
                self.conv.RecordDiagnostics(diag_prev_by_key)
                converged, info = self.conv.Commit(
                    completed_iter,
                    will_continue=(iter < itermax),
                )

                gcheck = info['cross']['GLoc-GImp']['abs']
                dG_iter = info['self']['GLoc']['abs']
                dmu = info['self']['mu']['abs']

                logger.info(
                    f"iteration : {completed_iter} | gcheck={gcheck:.3e} | "
                    f"dG={dG_iter:.3e}/dmu={dmu:.3e} | "
                    f"μ={green.mu}"
                )
            else:
                logger.info("iteration : 1 | convergence skipped")

            if converged:
                logger.info(
                    f"DMFT self-consistency achieved at iter {completed_iter}"
                )
                break
            elif iter == itermax:
                logger.info(
                    f"DMFT max iter {itermax} reached; "
                    f"gcheck={gcheck:.3e}, dG={dG_iter:.3e}, "
                    f"dmu={dmu:.3e}"
                )
            else:
                sigmah_current = sigc.sigh
                sigmaf_current = sigc.sigf
                sigc_current = sigc.sigimp
                green = green_next
                diag_prev_by_key = diag_by_key

            gc.collect()

    def EDMFT(self):

        errmessage = "missing input for EDMFT calculation"

        itermax = self.control["run"]["nscf"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'edmft'

        config = self.control.get("impurity", self.control.get("edmft", {}))
        impdict = config.get("impdict", config.get("ImpDict"))
        equiv = config.get("equiv", config.get("Equiv"))

        if impdict is None:
            raise KeyError("EDMFT requires control['impurity']['impdict']")
        if equiv is None:
            raise KeyError("EDMFT requires control['impurity']['equiv']")

        projector = Projector(
            basisindex=self.crystal._basis_index,
            impdict=copy.deepcopy(impdict),
            equiv=copy.deepcopy(equiv),
        )
        problem_keys = list(projector.fprojector.keys())
        if len(problem_keys) != 1:
            raise ValueError("EDMFT currently supports exactly one impurity problem")
        impurity_key = problem_keys[0]
        self.vbare.vloc.projector = projector
        if hasattr(self.vbare.vloc, "BuildProjection"):
            self.vbare.vloc.BuildProjection(projector)

        sigmah_current = None
        sigmaf_current = None
        sigc_current = None

        green = G(
            crystal=self.crystal,
            dlr=self.dlr,
            greenbare=self.greenbare.kf,
            hdf5file=hdf5file,
            group=group,
        )
        green.Save('gkf_ini', scf=False)

        mix = self.control["run"]["mix"]
        min_iter = self.conv.min_iter

        logger.info(f"[EDMFT] mix={mix}, min_iter={min_iter}")

        if itermax <= min_iter:
            logger.warning(
                f"nscf={itermax} <= min_iter={min_iter}; delta convergence "
                f"cannot trigger break. Loop will run to max_iter."
            )

        self.conv.Start()
        diag_prev_by_key = None

        # Bare bosonic lattice state used only to construct the first
        # impurity bath.  Subsequent W updates are made next to green_next.
        wlat = W(
            crystal=self.crystal,
            dlr=self.dlr,
            pol=None,
            vbare=self.vbare,
            c=self.c,
            hdf5file=hdf5file,
            group=group,
            iteration=0,
        )
        wlat.Save('wkf')

        for iter in range(1, itermax + 1):
            sigc = SigC(crystal=self.crystal, dlr=self.dlr)
            diag_by_key = {}

            for key in problem_keys:
                gloc = GLoc(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    green=green.kf,
                    key=key,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter - 1,
                    scf=True,
                )
                sighloc = None
                sigfloc = None
                sigcloc = None
                if sigmah_current is not None:
                    sighloc = eimp.Projection(sigmah_current, key)
                if sigmaf_current is not None:
                    sigfloc = eimp.Projection(sigmaf_current, key)
                if sigc_current is not None:
                    sigcloc = gloc.Projection(sigc_current, key)
                sigh_sample = None if sighloc is None else np.asarray(sighloc).flat[0]
                print(
                    f"[EDMFT Hyb-build] key={key}, "
                    f"sighloc[0]={sigh_sample}"
                )                
                eimp = EImp(
                    crystal=self.crystal,
                    projector=projector,
                    key=key,
                    hamtb=self.niham.k,
                    mu=green.mu,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                eimp.Save('eimp')

                hyb = Hyb(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    green=gloc.f,
                    eimp=eimp.e,
                    sigh=sighloc,
                    sigf=sigfloc,
                    sigc=sigcloc,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                hyb.Save('hyb')

                fweiss = FWeiss(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    eimp=eimp,
                    hyb=hyb,
                    mu=green.mu,
                    hdf5file=hdf5file,
                    group=group,
                )

                wloc = WLoc(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    wlat=wlat.kf,
                    vloc=self.vbare.vloc.vproj[key],
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter - 1,
                )
                wloc.Save('wloc')

                bweiss = BWeiss(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    vloc=self.vbare.vloc,
                    w=wloc,
                    p=None if iter == 1 else pimp,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                bweiss.Save('bweiss')

                impurity_method = ImpurityAction(
                    fweiss=fweiss,
                    bweiss=bweiss,
                    key=key,
                    control=self.control["run"],
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                impurity_result = impurity_method()

                for _attr in ("gimp", "sighimp", "sigfimp", "sigimp", "chi", "pimp", "wimp"):
                    if getattr(impurity_result, _attr, None) is None:
                        raise RuntimeError(
                            f"CTQMC produced no {_attr} for key={key}, iter={iter}. "
                            f"EDMFT requires dynamic impurity susceptibility output. "
                            f"Inspect params.obs.json, dyn.json, or solver stderr."
                        )

                sighimp = impurity_result.sighimp
                sigfimp = impurity_result.sigfimp
                sigimp = impurity_result.sigimp

                sigc.ImpEmbedding(
                    sigimp=sigimp.f,
                    sighimp=sighimp.h,
                    sigfimp=sigfimp.s,
                    projector=projector,
                    key=key,
                )
                pimp_new = impurity_result.pimp
                diag_by_key[key] = dict(impurity_result.diagnostics)

            sigc()

            green_next = G(
                crystal=self.crystal,
                dlr=self.dlr,
                greenbare=self.greenbare.kf,
                sigmagwc=sigc.kf,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
            )

            if not np.isfinite(green_next.mu):
                raise RuntimeError(
                    f"iter {iter}: mu solver produced non-finite mu={green_next.mu}. "
                    f"Likely cause: Sigma corruption (NaN/Inf) or bisection failure."
                )

            green_next.Save('gkf')

            polc = PolC(crystal=self.crystal, dlr=self.dlr)
            polc.ImpEmbedding(pimp_new, projector, impurity_key)
            polc()
            wlat_next = W(
                crystal=self.crystal,
                dlr=self.dlr,
                pol=polc.kf,
                vbare=self.vbare,
                c=self.c,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
            )
            wlat_next.Save('wkf')

            converged = False
            gcheck = float("nan")
            dG_iter = float("nan")
            dmu = float("nan")
            dW_iter = float("nan")
            if iter > 1:
                completed_iter = iter - 1
                self.conv.StartIter(completed_iter)
                self.conv.CheckSelfHDF5(
                    name="GLoc",
                    group=group,
                    subgroup="GLoc",
                    current=f"gloc.{completed_iter}",
                    previous=f"gloc.{completed_iter - 1}",
                    keys=problem_keys,
                )
                if iter > 2:
                    self.conv.CheckSelfHDF5(
                        name="PImp",
                        group=group,
                        subgroup="PImp",
                        current=f"pimp.{completed_iter}",
                        previous=f"pimp.{completed_iter - 1}",
                        keys=problem_keys,
                    )
                self.conv.CheckSelfHDF5(
                    name="WLoc",
                    group=group,
                    subgroup="WLoc",
                    current=f"wloc.{completed_iter}",
                    previous=f"wloc.{completed_iter - 1}",
                    keys=problem_keys,
                )
                self.conv.CheckCrossHDF5(
                    name_a="GLoc",
                    name_b="GImp",
                    group=group,
                    subgroup_a="GLoc",
                    subgroup_b="GImp",
                    stem_a=f"gloc.{completed_iter}",
                    stem_b=f"gimp.{completed_iter}",
                    keys=problem_keys,
                )
                self.conv.CheckSelf('mu', value=float(green.mu), kind='scalar')
                self.conv.CheckCrossHDF5(
                    name_a="WLoc",
                    name_b="WImp",
                    group=group,
                    subgroup_a="WLoc",
                    subgroup_b="WImp",
                    stem_a=f"wloc.{completed_iter}",
                    stem_b=f"wimp.{completed_iter}",
                    keys=problem_keys,
                )
                self.conv.RecordDiagnostics(diag_prev_by_key)
                converged, info = self.conv.Commit(
                    completed_iter,
                    will_continue=(iter < itermax),
                )

                gcheck = info.get('cross', {}).get('GLoc-GImp', {}).get('abs', float('nan'))
                dG_iter = info.get('self', {}).get('GLoc', {}).get('abs', float('nan'))
                dmu = info.get('self', {}).get('mu', {}).get('abs', float('nan'))
                dW_iter = info.get('self', {}).get('WLoc', {}).get('abs', float('nan'))

                logger.info(
                    f"iteration : {completed_iter} | gcheck={gcheck:.3e} | "
                    f"dG={dG_iter:.3e}/dW={dW_iter:.3e}/dmu={dmu:.3e} | "
                    f"μ={green.mu}"
                )
            else:
                logger.info("iteration : 1 | convergence skipped")

            # Keep only the terminal EDMFT result on the driver object.
            # Intermediate states are carried by local variables below.
            if converged or iter == itermax:
                self.green = green_next
                self.sigc = sigc
                self.w = wlat_next

            if converged:
                logger.info(
                    f"EDMFT self-consistency achieved at iter {completed_iter}"
                )
                break
            elif iter == itermax:
                logger.info(
                    f"EDMFT max iter {itermax} reached; "
                    f"gcheck={gcheck:.3e}, dG={dG_iter:.3e}, "
                    f"dW={dW_iter:.3e}, dmu={dmu:.3e}"
                )
            else:
                sigmah_current = sigc.sigh
                sigmaf_current = sigc.sigf
                sigc_current = sigc.sigimp
                green = green_next
                diag_prev_by_key = diag_by_key

                # CTQMC already mixed PImp in PostProcessing; these local
                # states are needed only when another iteration will run.
                pimp = pimp_new
                wlat = wlat_next

            gc.collect()

    def GWEDMFT(self):
        itermax = self.control["run"]["nscf"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'gwedmft'

        config = self.control.get("impurity", self.control.get("gwedmft", {}))
        impdict = config.get("impdict", config.get("ImpDict"))
        equiv = config.get("equiv", config.get("Equiv"))
        if impdict is None:
            raise KeyError("GW+EDMFT requires control['impurity']['impdict']")
        if equiv is None:
            raise KeyError("GW+EDMFT requires control['impurity']['equiv']")

        projector = Projector(
            basisindex=self.crystal._basis_index,
            impdict=copy.deepcopy(impdict),
            equiv=copy.deepcopy(equiv),
        )
        problem_keys = list(projector.fprojector.keys())
        if len(problem_keys) == 0:
            raise ValueError("GW+EDMFT requires at least one impurity problem")
        self.vbare.vloc.projector = projector
        if hasattr(self.vbare.vloc, "BuildProjection"):
            self.vbare.vloc.BuildProjection(projector)

        mix = self.control["run"]["mix"]
        mixing_method = self.control["run"]["mixing_method"]
        npulay = int(self.control["run"]["npulay"])
        min_iter = self.conv.min_iter
        logger.info(f"[GW+EDMFT] mix={mix}, min_iter={min_iter}")
        if itermax <= min_iter:
            logger.warning(
                f"nscf={itermax} <= min_iter={min_iter}; convergence cannot "
                f"trigger break. Loop will run to max_iter."
            )

        green = G(
            crystal=self.crystal,
            dlr=self.dlr,
            greenbare=self.greenbare.kf,
            hdf5file=hdf5file,
            group=group,
        )
        green.Save('gkf_ini', scf=False)
        wlat = W(
            crystal=self.crystal,
            dlr=self.dlr,
            pol=None,
            vbare=self.vbare,
            c=self.c,
            hdf5file=hdf5file,
            group=group,
            iteration=0,
        )
        wlat.Save('wkf_ini', scf=False)

        diag_prev_by_key = None
        polc_previous = None
        self.conv.Start()

        for iter in range(1, itermax + 1):
            hf = HF(
                occ=green.occ,
                occr=green.occr,
                v=self.vbare,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
                mix=mix,
                mixing_method=mixing_method,
                npulay=npulay,
            )
            hf_result = hf()
            gw = GW(
                g=green,
                w=wlat,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
                mix=mix,
                mixing_method=mixing_method,
                npulay=npulay,
            )
            gw_result = gw()
            sigc = SigC(
                crystal=self.crystal,
                dlr=self.dlr,
                sigh=hf_result.sigh.k,
                sigf=hf_result.sigf.k,
                siggwc=gw_result.siggwc.kf,
            )
            polc = PolC(crystal=self.crystal, dlr=self.dlr)
            polc.GWContribution(gw_result.pol)
            diag_by_key = {}

            for key in problem_keys:
                gloc = GLoc(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    green=green.kf,
                    key=key,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter - 1,
                    scf=True,
                )
                wloc = WLoc(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    wlat=wlat.kf,
                    vloc=self.vbare.vloc.vproj[key],
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter - 1,
                )
                wloc.Save('wloc')

                # Double Counting: subtract the local GW diagrams before the impurity calculation is added.
                hf_loc_result = HFLoc(
                    gloc=gloc,
                    vloc=self.vbare.vloc,
                    key=key,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )()
                gw_loc_result = GWLoc(
                    gloc=gloc,
                    wloc=wloc,
                    key=key,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )()

                # Green DC path: subtract the local GW diagrams before the
                # impurity calculation is added.  Keeping this separate from
                # the EDMFT contribution makes the assembly explicit:
                # Sigma = Sigma_GW - Sigma_DC + Sigma_EDMFT and
                # P = P_GW - P_DC + P_EDMFT.
                sigc.ImpEmbedding(
                    sigfimp=-hf_loc_result.sigf.floc,
                    sigimp=-gw_loc_result.siggwc.f,
                    projector=projector,
                    key=key,
                )
                polc.GWDoubleCounting(gw_loc_result.pol, projector, key)

                # Build the impurity Weiss fields from the projected lattice
                # quantities.  The bosonic Weiss field uses only the previous
                # impurity polarization; local GW polarization remains a
                # lattice double-counting term above.
                eimp = EImp(
                    crystal=self.crystal,
                    projector=projector,
                    key=key,
                    hamtb=self.niham.k,
                    sigh = hf_result.sigh.k,
                    sigf = hf_result.sigf.k,
                    mu=green.mu,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                eimp.Save('eimp')
                hyb = Hyb(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    green=gloc.f,
                    eimp=eimp.e,
                    sigh=hf_loc_result.sigh.Projection(hf_result.sigh.k, key),
                    sigf=hf_loc_result.sigf.Projection(hf_result.sigf.k, key),
                    sigc=gw_loc_result.siggwc.Projection(gw_result.siggwc.kf, key),
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                hyb.Save('hyb')
                fweiss = FWeiss(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    eimp=eimp,
                    hyb=hyb,
                    mu=green.mu,
                    hdf5file=hdf5file,
                    group=group,
                )
                p = None
                if polc_previous is not None:
                    p = gw_loc_result.pol.Projection(
                        polc_previous.pimp,
                        key,
                    )
                bweiss = BWeiss(
                    crystal=self.crystal,
                    dlr=self.dlr,
                    projector=projector,
                    key=key,
                    vloc=self.vbare.vloc,
                    w=wloc,
                    p=p,
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )
                bweiss.Save('bweiss')

                impurity_result = ImpurityAction(
                    fweiss=fweiss,
                    bweiss=bweiss,
                    key=key,
                    control=self.control["run"],
                    hdf5file=hdf5file,
                    group=group,
                    iteration=iter,
                )()
                for attr in (
                    "gimp", "sighimp", "sigfimp", "sigimp", "chi", "pimp", "wimp"
                ):
                    if getattr(impurity_result, attr, None) is None:
                        raise RuntimeError(
                            f"CTQMC produced no {attr} for key={key}, iter={iter}. "
                            f"GW+EDMFT requires the complete impurity pipeline output."
                        )

                sigfimp_new = impurity_result.sigfimp
                sigimp_new = impurity_result.sigimp
                pimp_new = impurity_result.pimp
                sigc.ImpEmbedding(
                    sigfimp=sigfimp_new.s,
                    sigimp=sigimp_new.f,
                    projector=projector,
                    key=key,
                )
                polc.ImpEmbedding(pimp_new, projector, key)

                diag_by_key[key] = dict(impurity_result.diagnostics)

            sigc()
            polc()

            green_next = G(
                crystal=self.crystal,
                dlr=self.dlr,
                greenbare=self.greenbare.kf,
                sigmagwc=sigc.kf,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
            )
            if not np.isfinite(green_next.mu):
                raise RuntimeError(
                    f"iter {iter}: mu solver produced non-finite mu={green_next.mu}."
                )
            green_next.Save('gkf')
            wlat_next = W(
                crystal=self.crystal,
                dlr=self.dlr,
                pol=polc.kf,
                vbare=self.vbare,
                c=self.c,
                hdf5file=hdf5file,
                group=group,
                iteration=iter,
            )
            wlat_next.Save('wkf')

            converged = False
            gcheck = dG_iter = dW_iter = dmu = float("nan")
            if iter > 1:
                completed_iter = iter - 1
                self.conv.StartIter(completed_iter)
                self.conv.CheckSelfHDF5(
                    name="GLoc", group=group, subgroup="GLoc",
                    current=f"gloc.{completed_iter}",
                    previous=f"gloc.{completed_iter - 1}", keys=problem_keys,
                )
                if iter > 2:
                    self.conv.CheckSelfHDF5(
                        name="PImp", group=group, subgroup="PImp",
                        current=f"pimp.{completed_iter}",
                        previous=f"pimp.{completed_iter - 1}", keys=problem_keys,
                    )
                self.conv.CheckSelfHDF5(
                    name="WLoc", group=group, subgroup="WLoc",
                    current=f"wloc.{completed_iter}",
                    previous=f"wloc.{completed_iter - 1}", keys=problem_keys,
                )
                self.conv.CheckCrossHDF5(
                    name_a="GLoc", name_b="GImp", group=group,
                    subgroup_a="GLoc", subgroup_b="GImp",
                    stem_a=f"gloc.{completed_iter}",
                    stem_b=f"gimp.{completed_iter}", keys=problem_keys,
                )
                self.conv.CheckCrossHDF5(
                    name_a="WLoc", name_b="WImp", group=group,
                    subgroup_a="WLoc", subgroup_b="WImp",
                    stem_a=f"wloc.{completed_iter}",
                    stem_b=f"wimp.{completed_iter}", keys=problem_keys,
                )
                self.conv.CheckSelf('mu', value=float(green.mu), kind='scalar')
                self.conv.RecordDiagnostics(diag_prev_by_key)
                converged, info = self.conv.Commit(
                    completed_iter, will_continue=(iter < itermax)
                )
                gcheck = info.get('cross', {}).get('GLoc-GImp', {}).get('abs', float('nan'))
                dG_iter = info.get('self', {}).get('GLoc', {}).get('abs', float('nan'))
                dW_iter = info.get('self', {}).get('WLoc', {}).get('abs', float('nan'))
                dmu = info.get('self', {}).get('mu', {}).get('abs', float('nan'))
                logger.info(
                    f"iteration : {completed_iter} | gcheck={gcheck:.3e} | "
                    f"dG={dG_iter:.3e}/dW={dW_iter:.3e}/dmu={dmu:.3e} | "
                    f"μ={green.mu}"
                )
            else:
                logger.info("iteration : 1 | convergence skipped")

            if converged or iter == itermax:
                self.green = green_next
                self.w = wlat_next
                self.pol = polc
                self.sigc = sigc

            if converged:
                logger.info(
                    f"GW+EDMFT self-consistency achieved at iter {completed_iter}"
                )
                break
            if iter == itermax:
                logger.info(
                    f"GW+EDMFT max iter {itermax} reached; gcheck={gcheck:.3e}, "
                    f"dG={dG_iter:.3e}, dW={dW_iter:.3e}, dmu={dmu:.3e}"
                )
            else:
                green = green_next
                wlat = wlat_next
                polc_previous = polc
                diag_prev_by_key = diag_by_key

            gc.collect()
