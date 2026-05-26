import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
import sys, time
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
from .CTQMC import CTQMC

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
        print(self.niham.k[:, :, 0, 0])

        self.vbare = V(crystal=self.crystal, twobody=control['ham'].get('twobody'), hdf5file=control["run"]["fn"]+'.h5', group='init')

        self.greenbare = G0(crystal=self.crystal, dlr=self.dlr, hamtb=self.niham.k, hdf5file=control["run"]["fn"]+'.h5', group='init')

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

        if (mode == 'FromScratch'):
            
            niham = self.niham
            vbare = self.vbare
            

        elif (mode == 'Restart'):
            group = group + '_restart'
            niham = self.niham
            vbare = self.vbare


        onebody = self.control["ham"].get("onebody")
        # twobody = self.control["ham"].get("twobody")


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
                    
                    

                hartreeold = None
                fockold = None

            logger.debug(hold.occ)
            sigmah = SigH(crystal=self.crystal,occ=hold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            sigmah.k = sigmah.Mixing(iter=iter,mix=mix,Fb=sigmah.k,Fm=hartreeold)
            if (iter % 50 == 0):
                sigmah.Save(f'sigh.{iter}')
            sigmaf = SigF(crystal=self.crystal,occr=hold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            sigmaf.k = sigmaf.Mixing(iter=iter,mix=mix,Fb=sigmaf.k,Fm=fockold)
            if (iter % 50 == 0):
                sigmaf.Save(f'sigf.{iter}')
            hnew = H(crystal=self.crystal,ham=niham.k,beta=self.dlr.beta,sigmah=sigmah.k,sigmaf=sigmaf.k,hdf5file=hdf5file,group=group)
            # hnew = Hamiltonian(crystal=self.crystal,ham=niham.k,beta=self.ft.beta,sigmah=None,sigmaf=sigmaf,hdf5file=fn,group=group)
            if (iter % 50 == 0):
                hnew.Save(f'hk.{iter}')

            converged, info = self.conv.Check(iter,
                fnew=hnew.k, fold=hold.k,
                mu_new=hnew.mu, mu_old=hold.mu)
            fcheck = info["fcheck"]
            logger.info(f"iteration : {iter}\ncriteria : {fcheck}\nchemical potential : {hnew.mu}")
            if converged:
                logger.info(f"Self-consistency is achived with {iter}-th")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
                break
            elif(iter==itermax):
                logger.warning(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.ham=hnew
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                hnew.Save('hk',True)
                sigmah.Save('sigh')
                sigmaf.Save('sigf')
                del hnew, sigmah, sigmaf, hold
                # del hnew, sigmaf, hold
                gc.collect()
            else:
                # hnew.OccMixing(iter=iter, mix=mix, occkb = hnew.occk, occkm=hold.occk)
                hold=hnew
                hartreeold = sigmah.k
                fockold = sigmaf.k
                del sigmaf,sigmah,hnew
                # del sigmaf,hnew
                gc.collect()


    def GWApproximation(self):

        errmessage = "missing input for GW calculation"
    
        itermax = self.control["run"]["nscf"]
        mix = self.control["run"]["mix"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        # mode = self.control["run"]["mode"]
        group = 'gw'

        niham = self.niham
        gbare = self.greenbare
        vbare = self.vbare

        pol_mixer = Mixing()
        sig_mixer = Mixing()

        self.gw_object_times = []
        for iter in range(1,itermax+1):
            iter_timing = {"iter": iter}
            if iter == 1:
                
                t0 = time.perf_counter()
                gold = G(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,hdf5file=hdf5file,group=group)
                logger.info(f"Initial chemical potential : {gold.mu}")
                iter_timing["GreenInt_init"] = time.perf_counter() - t0
                gold.Save(f'gkf_ini')
                wold = 0
                # gbare.Save('gbare')

            logger.info("Density Matrix :")
            logger.debug(gold.occ)
            # print("Hartree calculation start")
            sigmah = SigH(crystal=self.crystal,occ=gold.occ,vbare=vbare.k,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigmah.Save(f'sigmah.{iter}')
            # print("Hartree calculation finish")
            # print("Fock calculation start")
            sigmaf = SigF(crystal=self.crystal,occr=gold.occr,vbare=vbare.r,hdf5file=hdf5file,group=group)
            # if (iter % 50 == 0)or(iter == 1):
            sigmaf.Save(f'sigmaf.{iter}')
            # print("Fock calculation finish")
            # print("Polarizability calculation start")
            t0 = time.perf_counter()
            pol = P(crystal=self.crystal,dlr=self.dlr,green=gold.rt,hdf5file=hdf5file,group=group)
            iter_timing["Polarizability"] = time.perf_counter() - t0
            if iter == 1:
                pkfold = np.zeros_like(pol.kf)
            pol.kf = pol_mixer(iter=iter, mix=mix, Fnew=pol.kf, Fold=pkfold)
            pol.Save(f'pkf.{iter}')
            # print("Polarizability calculation finish")
            # print("Screened coulomb interaction calculation start")
            t0 = time.perf_counter()
            w = W(crystal=self.crystal,dlr=self.dlr,pol=pol.kf,vbare=vbare,c=self.c,hdf5file=hdf5file,group=group)
            iter_timing["WLat"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            w.Save(f'wkf.{iter}')
            # w.Save(w.ckf,f'wckf.{iter}')
            # print("Screened coulomb interaction calculation finish")
            # print("GW self-energy calculation start")
            t0 = time.perf_counter()
            sigmagwc = SigGWC(crystal=self.crystal,dlr=self.dlr,green=gold.rt,wlat=w.crt,hdf5file=hdf5file,group=group)
            iter_timing["SigmaGW"] = time.perf_counter() - t0
            if iter == 1:
                ckfold = np.zeros_like(sigmagwc.kf)
            sigmagwc.kf = sig_mixer(iter=iter, mix=mix, Fnew=sigmagwc.kf, Fold=ckfold)
            sigmagwc.Save(f'sigmagwckf.{iter}')
            # print("GW self-energy calculation finish")
            # print("GW green's function calculation start")
            t0 = time.perf_counter()

            gnew = G(crystal=self.crystal,dlr=self.dlr,greenbare=gbare.kf,sigmah=sigmah.k,sigmaf=sigmaf.k,sigmagwc=sigmagwc.kf,hdf5file=hdf5file,group=group)
            iter_timing["GreenInt"] = time.perf_counter() - t0
            # if (iter % 50 == 0)or(iter == 1):
            gnew.Save(f'gkf.{iter}')
            # print("GW green's function calculation start")
            self.gw_object_times.append(iter_timing)
            init_msg = ""
            if "GreenInt_init" in iter_timing:
                init_msg = f", GreenInt_init: {iter_timing['GreenInt_init']:.4f}s"
            logger.info(
                f"[GW timing][iter {iter}] GreenInt: {iter_timing['GreenInt']:.4f}s, "
                f"Polarizability: {iter_timing['Polarizability']:.4f}s, "
                f"WLat: {iter_timing['WLat']:.4f}s, "
                f"SigmaGW: {iter_timing['SigmaGW']:.4f}s{init_msg}"
            )
            converged, info = self.conv.Check(iter,
                fnew=gnew.kf, fold=gold.kf,
                mu_new=gnew.mu, mu_old=gold.mu,
                bnew=w.kf, bold=wold,
                npulay=pol_mixer.npulay)
            fcheck = info["fcheck"]
            bcheck = info["bcheck"]

            logger.info(f"iteration : {iter} \nfcriteria : {fcheck} \nbcriteria : {bcheck} \nchemicalpotential : {gnew.mu}")

            if converged:
                logger.info(f"Self-consistency is achived with {iter}-th")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
                break
            elif (iter==itermax):
                logger.info(f"Notice: Broadening schemes will be turned off from the {iter}-th iteration.")
                self.green = gnew
                self.pol = pol
                self.w = w
                self.sigmagwc = sigmagwc
                self.sigmaf = sigmaf
                self.sigmah = sigmah
                gnew.Save('gkf',chem=True)
                sigmah.Save('sigmah')
                sigmaf.Save('sigmaf')
                sigmagwc.Save('sigmagwckf')
                pol.Save('pkf')
                w.Save('wkf')
                # self.sigmagwc.SigmaStc()
                # self.sigmagwc.Zfactor()
                del niham, vbare, gbare, gnew, gold, sigmaf, sigmah, sigmagwc, pol, w
                gc.collect()
            else:
                gold = gnew
                ckfold = sigmagwc.kf
                pkfold = pol.kf
                wold = w.kf

                del gnew, sigmah, sigmaf, sigmagwc, pol, w
                gc.collect()

    def ImpurityAction(self):

        errmessage = "missing input for DMFT calculation"

        itermax = self.control["run"]["nscf"]
        hdf5file = self.control["run"]["fn"] + '.h5'
        group = 'impurity_solver'
        # self.control["run"]["method"]

        config = self.control.get("impurity", self.control.get("dmft", {}))
        impdict = config.get("impdict", config.get("ImpDict"))
        equiv = config.get("equiv", config.get("Equiv"))

        if impdict is None:
            raise KeyError("DMFT requires control['impurity']['impdict']")
        if equiv is None:
            raise KeyError("DMFT requires control['impurity']['equiv']")

        projector = Projector(basisindex=self.crystal._basis_index,impdict=copy.deepcopy(impdict),equiv=copy.deepcopy(equiv),)
        self.vbare.vloc.projector=projector
        voption = self.control["ham"]["twobody"].get("Local")
        if voption is None:
            raise KeyError("DMFT requires control['ham']['twobody']['Local']")
        # vloc = VLoc(crystal=self.crystal, voption=voption)
        

        sigmah_current = None
        sigmaf_current = None
        sigc_current = None

        t0 = time.perf_counter()
        green = G(
            crystal=self.crystal,
            dlr=self.dlr,
            greenbare=self.greenbare.kf,
            hdf5file=hdf5file,
            group=group,
        )
        initial_green_time = time.perf_counter() - t0
        green.Save(f'gkf_ini')

        t0 = time.perf_counter()
        gloc = GLoc(crystal=self.crystal,dlr=self.dlr,projector=projector,green=green.kf,hdf5file=hdf5file,group=group,)
        initial_gloc_time = time.perf_counter() - t0
        gloc.Save(f'gloc_ini')

        self.dmft_object_times = []

        # ---- Convergence configuration owned by self.conv; driver still
        #      reads mix_sigma (mixing is not convergence logic) and the
        #      causality tolerances (driver calls causality checks directly).
        mix = float(self.control["run"].get("mix_sigma", 0.1))
        causality_tol_hyb = self.conv.causality_tol_hyb
        causality_tol_sig = self.conv.causality_tol_sig
        min_iter = self.conv.min_iter

        logger.info(f"[ImpurityAction] mix_sigma={mix}, min_iter={min_iter}")

        if itermax <= min_iter:
            logger.warning(f"nscf={itermax} <= min_iter={min_iter}; delta convergence "
                           f"cannot trigger break. Loop will run to max_iter.")

        self.conv.Start()
        sig_cache_by_key = {}

        for iter in range(1, itermax+1):
            iter_timing = {
                "iter": iter,
                "G": initial_green_time if iter == 1 else 0.0,
                "GLoc": initial_gloc_time if iter == 1 else 0.0,
            }

            sigctemp = np.zeros_like(green.kf)
            sightemp = np.zeros_like(self.niham.k)
            sigftemp = np.zeros_like(self.niham.k)

            gcheck = 0.0
            iter_timing["BWeiss"] = 0.0
            iter_timing["FWeiss"] = 0.0
            iter_timing["CTQMC"] = 0.0
            if self.control["run"]["method"] == "gw+edmft":
                subtract_local_dc = True
            else:
                subtract_local_dc = False
            
            gimp_by_key = {}
            sig_rolled_back = []   # per-iter list of keys where Σ causality rolled back
            diag_by_key = {}        # captured immediately inside the per-key loop

            for key in projector.fprojector.keys():
                norbc = projector.fprojector[key].shape[1]
                dc_shape = (norbc, norbc, self.crystal.ns)
                sigh_dc = np.zeros(dc_shape, dtype=np.complex128, order='F')
                sigf_dc = np.zeros(dc_shape, dtype=np.complex128, order='F')

                if subtract_local_dc:
                    sigh_dc_obj = SigHLoc(crystal=self.crystal, projector=projector, occ=gloc.occ[key], vloc=self.vbare.vloc.Projection(self.vbare.vloc.vloc, key=key), key=key, hdf5file=hdf5file, group=group)
                    sigf_dc_obj = SigFLoc(crystal=self.crystal, projector=projector, occ=gloc.occ[key], vloc=self.vbare.vloc.Projection(self.vbare.vloc.vloc, key=key), key=key, hdf5file=hdf5file, group=group)
                    sigh_dc = sigh_dc_obj.hloc
                    sigf_dc = sigf_dc_obj.floc
                    sigh_dc_obj.Save(f'sigh_dc.{iter}.{key}')
                    sigf_dc_obj.Save(f'sigf_dc.{iter}.{key}')
                t0 = time.perf_counter()
                eimp = EImp(crystal=self.crystal,projector=projector,key=key,hamtb=self.niham.k,mu=green.mu, hloc=sigh_dc, floc=sigf_dc, hdf5file=hdf5file, group=group)
                eimp.Save(f'eimp.{iter}.{key}')

                sighloc = None
                sigfloc = None
                sigcloc = None
                if sigmah_current is not None:
                    sighloc = eimp.Projection(sigmah_current, key)
                    # sighloc carries no c-shift: lattice G no longer subtracts Hartree trace.
                if sigmaf_current is not None:
                    sigfloc = eimp.Projection(sigmaf_current, key)
                if sigc_current is not None:
                    sigcloc = gloc.Projection(sigc_current, key)
                print(f"[Hyb-build] key={key}, sighloc[0,0,0]={None if sighloc is None else sighloc[0,0,0]}")

                hyb = Hyb(crystal=self.crystal,dlr=self.dlr,projector=projector,key=key,green=gloc.f[key],eimp=eimp.e,sigh=sighloc,sigf=sigfloc,sigc=sigcloc,hdf5file=hdf5file,group=group)
                hyb.Save(f'hyb.{iter}.{key}')

                fweiss = FWeiss(
                    crystal=self.crystal, dlr=self.dlr, projector=projector,
                    key=key, eimp=eimp, hyb=hyb, mu=green.mu,
                    hdf5file=hdf5file, group=group,
                )

                # ---- Hyb causality (hard gate, comdmft.py:920-922 pattern) ----
                #      Positive-only uniform grid; caller raises after logging.
                ok_hyb, max_im_hyb = fweiss.CheckCausality(causality_tol_hyb)
                if not ok_hyb:
                    self.conv.RecordCausalityFail(iter, kind="hyb", key=key,
                        max_im=max_im_hyb, mu=float(green.mu))
                    raise RuntimeError(
                        f"Hyb causality violated at key={key}, iter={iter}: "
                        f"max diagonal Im(Delta_uniform)={max_im_hyb:.3e} > "
                        f"tol={causality_tol_hyb:.1e}. "
                        f"comdmft.py:920-922 policy (hard exit)."
                    )
                iter_timing["FWeiss"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                bweiss = BWeiss(crystal=self.crystal,dlr=self.dlr,projector=projector,key=key,vloc=self.vbare.vloc,ploc=None,wloc=None,hdf5file=hdf5file,group=group,)
                bweiss.Save(f'bweiss.{iter}.{key}')
                iter_timing["BWeiss"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                ## Impurity Solver
                ctqmc = CTQMC(dlr=self.dlr,fweiss=fweiss,bweiss=bweiss,key=key,control=self.control["run"],hdf5file=hdf5file,group=group,)
                ctqmc.PreProcessing(iter=iter)
                ctqmc.Run(iter=iter)
                ctqmc.PostProcessing(iter=iter)
                
                iter_timing["CTQMC"] += time.perf_counter() - t0

                # ---- CTQMC output guard (F12): explicit failure if PostProcessing
                #      produced no sigma/Green's function for this key. ----
                for _attr in ("gimp", "sighimp", "sigfimp", "sigimp"):
                    if getattr(ctqmc, _attr, None) is None:
                        raise RuntimeError(
                            f"CTQMC produced no {_attr} for key={key}, iter={iter}. "
                            f"Inspect params.obs.json or solver stderr."
                        )

                gimp_new = ctqmc.gimp.f
                ctqmc.gimp.Save(f'gimp.{iter}.{key}')
                ctqmc.sighimp.Save(f'sighimp.{iter}.{key}')
                ctqmc.sigfimp.Save(f'sigfimp.{iter}.{key}')
                ctqmc.sigimp.Save(f'sigimp.{iter}.{key}')
                if (self.control["run"]["method"] == 'edmft')or(self.control["run"]["method"] ==("gw+edmft")):
                    if ctqmc.chi is not None and ctqmc.chi.f is not None:
                        ctqmc.chi.Save(f'chi.{iter}.{key}')
                    if ctqmc.pimp is not None and ctqmc.pimp.f is not None:
                        ctqmc.pimp.Save(f'pimp.{iter}.{key}')

                # ---- Sigma causality on positive-only uniform grid ----
                ok_sig, max_im_sig = ctqmc.sigimp.CheckCausalityUniform(causality_tol_sig)

                def _embed_from_impurity(sighimp_h, sigfimp_s, sigimp_f):
                    """Re-embed impurity-space Sigma into lattice using the
                    CURRENT iter's dc subtraction (F5: avoid stale dc on rollback)."""
                    sigc_e = green.Embedding(sigimp_f, projector=projector, key=key)
                    sigh_e = self.niham.Embedding(sighimp_h - sigh_dc, projector=projector, key=key)
                    sigf_e = self.niham.Embedding(sigfimp_s - sigf_dc, projector=projector, key=key)
                    return sigh_e, sigf_e, sigc_e

                if ok_sig:
                    sigh_embed, sigf_embed, sigc_embed = _embed_from_impurity(
                        ctqmc.sighimp.h, ctqmc.sigfimp.s, ctqmc.sigimp.f
                    )
                    # Cache impurity-space raw values (not embedded), so rollback
                    # can re-embed with the iter's current dc (F5). Also cache the
                    # accepted gimp so Gate-A-style diagnostic stays generation
                    # consistent (F4).
                    sig_cache_by_key[key] = {
                        "sighimp_h": ctqmc.sighimp.h.copy(),
                        "sigfimp_s": ctqmc.sigfimp.s.copy(),
                        "sigimp_f":  ctqmc.sigimp.f.copy(),
                        "gimp_f":    gimp_new.copy(),
                    }
                    gimp_by_key[key] = gimp_new
                else:
                    if key not in sig_cache_by_key:
                        if iter <= 2:
                            # Warmup grace: accept the raw value but DO NOT cache
                            # (caching a violating value would break rollback).
                            logger.warning(
                                f"iter {iter}: Sigma causality borderline at key={key} "
                                f"(Im={max_im_sig:.3e}), no cache yet -- accepting raw "
                                f"value as warmup (NOT cached)."
                            )
                            sigh_embed, sigf_embed, sigc_embed = _embed_from_impurity(
                                ctqmc.sighimp.h, ctqmc.sigfimp.s, ctqmc.sigimp.f
                            )
                            gimp_by_key[key] = gimp_new
                            sig_rolled_back.append(f"{key}(warmup,Im={max_im_sig:.3e})")
                        else:
                            self.conv.RecordCausalityFail(iter, kind="sig", key=key,
                                max_im=max_im_sig, mu=float(green.mu),
                                subreason="no-cache",
                                diag=getattr(ctqmc, "diagnostics", {}))
                            raise RuntimeError(
                                f"Sigma causality violated at key={key}, iter={iter} "
                                f"(max Im={max_im_sig:.3e}), no cached good value to "
                                f"roll back to. Increase min_iter, lower mix_sigma, "
                                f"or investigate CTQMC sampling."
                            )
                    else:
                        # Rollback path: re-embed cached impurity-space values
                        # with the CURRENT iter's dc subtraction (F5 fix).
                        cached = sig_cache_by_key[key]
                        sigh_embed, sigf_embed, sigc_embed = _embed_from_impurity(
                            cached["sighimp_h"], cached["sigfimp_s"], cached["sigimp_f"]
                        )
                        # Pin gimp_by_key to the same generation as the rolled-back
                        # Sigma (F4: Gate-A diagnostic stays apples-to-apples).
                        gimp_by_key[key] = cached["gimp_f"]
                        sig_rolled_back.append(f"{key}(rollback,Im={max_im_sig:.3e})")
                diag_by_key[key] = dict(getattr(ctqmc, "diagnostics", {}))

                sigctemp += sigc_embed
                sightemp += sigh_embed
                sigftemp += sigf_embed

            if iter == 1:
                sigmah_next = sightemp.copy()
                sigmaf_next = sigftemp.copy()
                sigc_next = sigctemp.copy()
            else:
                sigmah_next = mix * sightemp + (1.0 - mix) * sigmah_current
                sigmaf_next = mix * sigftemp + (1.0 - mix) * sigmaf_current
                sigc_next = mix * sigctemp + (1.0 - mix) * sigc_current

            t0 = time.perf_counter()
            green_next = G(
                crystal=self.crystal,
                dlr=self.dlr,
                greenbare=self.greenbare.kf,
                sigmah=sigmah_next,
                sigmaf=sigmaf_next,
                sigmagwc=sigc_next,
                hdf5file=hdf5file,
                group=group,
            )
            iter_timing["G"] += time.perf_counter() - t0

            # ---- mu NaN guard (F10): catch corrupted chemical potential
            #      before it propagates into the next iter's G. ----
            if not np.isfinite(green_next.mu):
                raise RuntimeError(
                    f"iter {iter}: mu solver produced non-finite mu={green_next.mu}. "
                    f"Likely cause: Sigma corruption (NaN/Inf) or bisection failure."
                )

            t0 = time.perf_counter()
            gloc_next = GLoc(crystal=self.crystal,dlr=self.dlr,projector=projector,green=green_next.kf,hdf5file=hdf5file,group=group,)
            iter_timing["GLoc"] += time.perf_counter() - t0
            green_next.Save(f'gkf.{iter}')
            gloc_next.Save(f'gloc.{iter}')

            # ---- Convergence step (gcheck + delta + row write + prev snapshot) ----
            converged, info = self.conv.CheckDelta(iter,
                gloc_next=gloc_next,
                sigmah_next=sigmah_next, sigmaf_next=sigmaf_next, sigc_next=sigc_next,
                mu_next=float(green_next.mu),
                gimp_by_key=gimp_by_key, diag_by_key=diag_by_key,
                sig_rolled_back=sig_rolled_back,
                will_continue=(iter < itermax))
            gcheck = info["gcheck"]
            dG_iter = info["dG_iter"]
            dSig_iter = info["dSig_iter"]
            dmu = info["dmu"]

            # Per-key raw diagnostics → HDF5 for post-analysis
            try:
                with h5py.File(self.control["run"]["fn"] + ".h5", "a") as h:
                    g = h.require_group(f"impurity_solver/diag/iter_{iter}")
                    for k, d in diag_by_key.items():
                        sub = g.require_group(str(k))
                        for nm, val in d.items():
                            if nm == "histo":
                                if nm in sub:
                                    del sub[nm]
                                sub.create_dataset(nm, data=np.asarray(val))
                            else:
                                sub.attrs[nm] = float(val) if val is not None else float("nan")
            except Exception as e:
                logger.warning(f"per-key diag HDF5 write failed: {e}")

            self.dmft_object_times.append(iter_timing)
            logger.info(
                f"[DMFT timing][iter {iter}] G: {iter_timing['G']:.4f}s, "
                f"GLoc: {iter_timing['GLoc']:.4f}s, "
                f"BWeiss: {iter_timing['BWeiss']:.4f}s, "
                f"FWeiss: {iter_timing['FWeiss']:.4f}s, "
                f"CTQMC: {iter_timing['CTQMC']:.4f}s"
            )
            logger.info(
                f"iteration : {iter} | gcheck={gcheck:.3e} | "
                f"dG={dG_iter:.3e}/dSig={dSig_iter:.3e}/dmu={dmu:.3e} | "
                f"μ={green_next.mu}"
            )

            if converged:
                if info["gcheck_warn_triggered"]:
                    logger.warning(
                        f"DMFT delta-converged at iter {iter} but gcheck={gcheck:.3e} "
                        f"exceeds sanity threshold ({info['gcheck_warn_threshold']:.1e}). "
                        f"Self-consistency between lattice and impurity G may be "
                        f"incomplete; inspect convergence.log."
                    )
                logger.info(f"DMFT self-consistency achieved at iter {iter}")
                break
            elif iter == itermax:
                logger.info(
                    f"DMFT max iter {itermax} reached; "
                    f"gcheck={gcheck:.3e}, dG={dG_iter:.3e}, "
                    f"dSig={dSig_iter:.3e}, dmu={dmu:.3e}"
                )
            else:
                sigmah_current = sigmah_next
                sigmaf_current = sigmaf_next
                sigc_current   = sigc_next
                green = green_next
                gloc  = gloc_next

            gc.collect()
