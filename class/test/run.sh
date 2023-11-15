#!/bin/zsh
python3 -u TestCommon.py >> result.log1
python3 -u TestDyson.py >> result.log2
python3 -u TestEmbedding.py >> result.log3
python3 -u TestProjection.py >> result.log4
python3 -u TestFourierFT.py >> result.log5
python3 -u TestFourierKR.py >> result.log6
