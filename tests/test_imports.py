import re
from pathlib import Path


def test_core_public_imports():
    import QAssemble
    from QAssemble import (
        BLatStc,
        Crystal,
        DLR,
        G,
        G0,
        H,
        H0,
        P,
        Run,
        SigF,
        SigGWC,
        SigH,
        V,
        W,
    )

    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert QAssemble.__version__ == re.search(r'^version = "([^"]+)"$', pyproject, re.M).group(1)
    for obj in (BLatStc, Crystal, DLR, G, G0, H, H0, P, Run, SigF, SigGWC, SigH, V, W):
        assert obj is not None


def test_utility_namespaces_exposed():
    import QAssemble
    from QAssemble.utility.Bare import Bare
    from QAssemble.utility.Common import Common
    from QAssemble.utility.Dyson import Dyson
    from QAssemble.utility.Embedding import Embedding
    from QAssemble.utility.Fourier import Fourier
    from QAssemble.utility.Projection import Projection

    for name in ("Bare", "Common", "Dyson", "Embedding", "Fourier", "Projection"):
        assert getattr(QAssemble, name) is locals()[name]

    assert callable(Dyson.FLatStc)
    assert callable(Bare.FLatFreq)
    assert callable(Common.MatInv)
    assert Dyson.FLatStc is not Projection.FLatStc
    assert Dyson.FLatStc is not Embedding.FLatStc
    assert Embedding.FLatStc is not Projection.FLatStc
