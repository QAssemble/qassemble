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

    assert QAssemble.__version__
    for obj in (BLatStc, Crystal, DLR, G, G0, H, H0, P, Run, SigF, SigGWC, SigH, V, W):
        assert obj is not None
