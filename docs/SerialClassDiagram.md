# Serial Class Diagram

The following diagrams capture the main classes and composition relationships in the serial pipeline (`src/QuantumAssemble.py` and `src/QAssemble/Serial`). Solid arrows point from an owner to the component it instantiates or calls directly. Dashed arrows highlight helper utilities that are passed in or referenced for numerical work.

## High-Level Flow

```mermaid
classDiagram
    direction LR
    class Run {
        +control: dict
        +func: CorrelationFunction
        +ReadInput()
        +CheckKeyinString()
        +RunDiagE()
    }

    class CorrelationFunction {
        +crystal: Crystal
        +dlr: DLR
        +TightBinding()
        +HartreeFock()
        +GWApproximation()
    }

    class Crystal {
        +kpoint
        +basisf
        +probspace
        +FAtomOrb()
        +OrbSpin2Composite()
        +Kpath()
    }

    class DLR {
        +omega
        +nu
        +tauF
        +FT2F()
        +BF2T()
    }


    Run --> CorrelationFunction : builds
    Run --> Crystal : via CorrelationFunction
    CorrelationFunction --> Crystal : owns
    CorrelationFunction --> DLR : owns
    
```

## Fermionic Stack

```mermaid
classDiagram
    direction LR
    class FLatDyn {
        +crystal: Crystal
        +dlr: DLR
        +F2T()
        +K2R()
        +Moment()
    }
    class FLatStc {
        +crystal: Crystal
        +Inverse()
        +Band()
        +DOS()
    }
    class GreenBare {
        +Cal()
        +Save()
    }
    class GreenInt {
        +Occ()
        +SearchMu()
        +UpdateMu()
    }
    class SigmaGWC {
        +Cal()
        +SigmaStc()
        +Zfactor()
    }
    class Hamiltonian {
        +CalMu0()
        +NumOfE()
        +SearchMu()
        +OccMixing()
    }
    class SigmaHartree {
        +Cal()
        +Mixing()
    }
    class SigmaFock {
        +Cal()
        +Mixing()
    }
    class FPathDyn {
        +Inverse()
        +R2K()
        +KArb()
    }
    class FPathStc {
        +Inverse()
        +R2K()
        +Slab()
        +Band()
    }

    CorrelationFunction --> FLatDyn
    CorrelationFunction --> FLatStc
    CorrelationFunction --> FPathDyn
    CorrelationFunction --> FPathStc
    FLatDyn --> Crystal
    FLatDyn --> DLR
    FLatStc --> Crystal
    FPathDyn --> Crystal
    FPathDyn ..> DLR
    FPathStc --> Crystal
    CorrelationFunction --> GreenBare
    CorrelationFunction --> GreenInt
    CorrelationFunction --> SigmaGWC
    CorrelationFunction --> Hamiltonian
    CorrelationFunction --> SigmaHartree
    CorrelationFunction --> SigmaFock
    GreenBare --> FLatDyn : writes
    GreenInt --> FLatDyn : consumes
    SigmaGWC --> FLatDyn : consumes
    Hamiltonian --> FLatStc : builds
    SigmaHartree --> FLatStc : uses
    SigmaFock --> FLatStc : uses
```

## Bosonic Stack

```mermaid
classDiagram
    direction LR
    class BLatDyn {
        +crystal: Crystal
        +dlr: DLR
        +Inverse()
        +Moment()
        +K2R()
    }
    class BLatStc {
        +crystal: Crystal
        +Inverse()
        +Mixing()
        +Dyson()
    }
    class VBare {
        +Cal()
        +LocPlusNonLoc()
        +OhnoYukawa()
    }
    class PolLat {
        +Cal()
        +Save()
    }
    class WLat {
        +Cal()
        +Save()
    }
    class BPathDyn {
        +R2K()
    }
    class BPathStc {
        +R2K()
        +Slab()
    }
    class BLocStc {
        +Imp2Loc()
        +Loc2Imp()
        +Arr2Dict()
    }
    class VLoc {
        +SetLocalInteracting()
        +SlaterKanamori()
        +AngularIntegral()
    }

    CorrelationFunction --> BLatDyn
    CorrelationFunction --> BLatStc
    CorrelationFunction --> BPathDyn
    CorrelationFunction --> BPathStc
    CorrelationFunction --> BLocStc
    CorrelationFunction --> VBare
    BLatDyn --> Crystal
    BLatDyn --> DLR
    BLatDyn ..> PolLat
    BLatDyn ..> WLat
    BLatStc --> Crystal
    VBare --> Crystal
    BPathDyn --> Crystal
    BPathStc --> Crystal
    BLocStc --> Crystal
    BLocStc ..> VLoc
```

## Shared Utilities

```mermaid
classDiagram
    direction LR
    class Fourier {
        +FLatDynK2R()
        +BLatStcR2K()
        +FPathDynR2K()
    }
    class Dyson {
        +FLatDyn()
        +FLatStc()
        +BLatDyn()
        +BLatStc()
    }
    class Bare {
        +FFreq()
        +BFreq()
        +FTau()
        +BTau()
    }
    class Common {
        +MatInv()
        +HermitianEigenCmplx()
        +SplineCmplx()
    }

    FLatDyn ..> Fourier
    FLatStc ..> Fourier
    FPathDyn ..> Fourier
    BLatDyn ..> Fourier
    BLatStc ..> Fourier
    BPathDyn ..> Fourier
    BPathStc ..> Fourier
    FLatDyn ..> Dyson
    FLatStc ..> Dyson
    BLatDyn ..> Dyson
    BLatStc ..> Dyson
    CorrelationFunction ..> Bare
    CorrelationFunction ..> Common
```

Use the diagrams alongside `docs/SerialModules.md` for deeper descriptions of each class and the numerical responsibilities they own.
