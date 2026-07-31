# Serial Class Diagram

The following diagrams capture the main classes and composition relationships in the serial pipeline (`src/QAssemble/`). Solid arrows point from an owner to the component it instantiates or calls directly. Dashed arrows highlight helper utilities that are passed in or referenced for numerical work.

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
        +Dict2Hdf5()
        +Hdf52Dict()
        +CheckInput()
        +ChangeInput()
        +CompareDict()
    }

    class CorrelationFunction {
        +crystal: Crystal
        +dlr: DLR
        +TightBinding()
        +HartreeFock()
        +GWApproximation()
        +SCFCheck()
    }

    class Crystal {
        +kpoint
        +basisf
        +probspace
        +FAtomOrb()
        +BAtomOrb()
        +OrbSpin2Composite()
        +Kpath()
        +KPoint()
        +MappingRVec()
        +Projector()
    }

    class DLR {
        +omega
        +nu
        +tauF
        +FT2F()
        +BF2T()
        +TauDLR2Uniform()
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
        +T2F()
        +K2R()
        +R2K()
        +Moment()
        +Inverse()
        +Mixing()
        +Dyson()
        +ChemEmbedding()
        +StcEmbedding()
        +Spectral()
        +R2KArb()
        +KArb()
        +Diagonalize()
    }
    class FLatStc {
        +crystal: Crystal
        +Inverse()
        +K2R()
        +R2K()
        +Band()
        +DOS()
        +Visualization()
        +Mixing()
        +Dyson()
        +ChemEmbedding()
        +R2KArb()
        +Diagonalize()
        +KValley()
    }
    class G0 {
        +Cal()
        +Save()
    }
    class G {
        +CalMu0()
        +Occ()
        +SearchMu()
        +UpdateMu()
        +NumOfE()
        +Save()
    }
    class SigGWC {
        +Cal()
        +SigStc()
        +Zfactor()
        +Save()
    }
    class H0 {
        +Cal()
        +Save()
        +Valley()
        +AntiValley()
    }
    class H {
        +CalMu0()
        +NumOfE()
        +SearchMu()
        +Occ()
        +UpdateMu()
        +OccMixing()
        +Save()
    }
    class SigH {
        +Cal()
        +Save()
    }
    class SigF {
        +Cal()
        +Save()
    }
    class Z {
        +Cal()
        +Save()
    }
    class SigStc {
        +Cal()
        +Save()
    }
    class GreenAB {
        +KI2KF()
    }
    class HamiltonianAB {
        +KI2KF()
    }
    class FPathDyn {
        +Inverse()
        +R2K()
        +KArb()
        +MQEMWrapper()
        +Spectral()
        +MQEMPrepare()
    }
    class FPathStc {
        +Inverse()
        +R2K()
        +K2R()
        +Slab()
        +SlabZmat()
        +Band()
        +Dos()
        +FermiSurface()
        +Occ()
        +Moments()
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
    CorrelationFunction --> G0
    CorrelationFunction --> G
    CorrelationFunction --> SigGWC
    CorrelationFunction --> H0
    CorrelationFunction --> H
    CorrelationFunction --> SigH
    CorrelationFunction --> SigF
    G0 --|> FLatDyn
    G --|> FLatDyn
    SigGWC --|> FLatDyn
    GreenAB --|> FLatDyn
    H0 --|> FLatStc
    H --|> FLatStc
    SigH --|> FLatStc
    SigF --|> FLatStc
    HamiltonianAB --|> FLatStc
    Z --|> FLatStc
    SigStc --|> FLatStc
    G0 --> FLatDyn : writes
    G --> FLatDyn : consumes
    SigGWC --> FLatDyn : consumes
    H0 --> FLatStc : builds
    H --> FLatStc : builds
    SigH --> FLatStc : uses
    SigF --> FLatStc : uses
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
        +F2T()
        +T2F()
        +K2R()
        +R2K()
        +Mixing()
        +Dyson()
        +StcEmbedding()
        +RT2mRmT()
        +TauF2TauB()
        +R2KArb()
        +Save()
    }
    class BLatStc {
        +crystal: Crystal
        +Inverse()
        +K2R()
        +R2K()
        +Mixing()
        +Dyson()
        +Save()
        +R2KArb()
        +HermitianCheck()
    }
    class V {
        +Cal()
        +LocPlusNonLoc()
        +OhnoYukawa()
        +OhnoParameter()
        +JTHPotential()
        +Save()
    }
    class P {
        +Cal()
        +Save()
    }
    class W {
        +Cal()
        +Save()
    }
    class BPathDyn {
        +R2K()
    }
    class BPathStc {
        +R2K()
    }
    class BLocStc {
        +Inverse()
        +Mixing()
        +Imp2Loc()
        +Loc2Imp()
        +Arr2Dict()
        +Dict2Arr()
        +Dyson()
        +Save()
    }
    class VLoc {
        +SetLocalInteracting()
        +GenOnsite()
        +SlaterKanamori()
        +SlaterParameter()
        +KanamoriParameter()
        +AngularIntegral()
        +RotationMatrix()
        +Spherical2Cubic()
        +GetUijklComCTQMC()
    }

    CorrelationFunction --> BLatDyn
    CorrelationFunction --> BLatStc
    CorrelationFunction --> BPathDyn
    CorrelationFunction --> BPathStc
    CorrelationFunction --> BLocStc
    CorrelationFunction --> V
    BLatDyn --> Crystal
    BLatDyn --> DLR
    P --|> BLatDyn
    W --|> BLatDyn
    BLatStc --> Crystal
    V --|> BLatStc
    V --> Crystal
    BPathDyn --> Crystal
    BPathStc --> Crystal
    BLocStc --> Crystal
    VLoc --|> BLocStc
```

## Shared Utilities

```mermaid
classDiagram
    direction LR
    class Fourier {
        +FLatDynK2R()
        +FLatStcK2R()
        +FLatDynR2K()
        +FLatStcR2K()
        +BLatDynK2R()
        +BLatStcK2R()
        +BLatDynR2K()
        +BLatStcR2K()
        +FPathStcR2K()
        +FPathDynR2K()
        +FLatDynM()
        +BLatDynM()
    }
    class Dyson {
        +FLatDyn()
        +FLatStc()
        +BLatDyn()
        +BLatStc()
        +FLocDyn()
        +FLocStc()
        +BLocDyn()
        +BLocStc()
    }
    class Bare {
        +FFreq()
        +BFreq()
        +FTau()
        +BTau()
        +FLocFreq()
        +FLatFreq()
        +BLocFreq()
        +BLatFreq()
    }
    class Common {
        +MatInv()
        +HermitianEigenCmplx()
        +SplineCmplx()
        +FderivCmplx()
        +BernoulliPolynomial()
        +EulerPolynomial()
        +MinDistance()
    }

    class Mixing {
        +method: str
        +npulay: int
        +reset()
        +__call__(iter, mix, Fnew, Fold)
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
    BLocStc ..> Dyson
    CorrelationFunction ..> Bare
    CorrelationFunction ..> Common
    CorrelationFunction ..> Mixing
```

Use the diagrams alongside `docs/SerialModules.md` for deeper descriptions of each class and the numerical responsibilities they own.
