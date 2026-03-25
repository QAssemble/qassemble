#!/usr/bin/env python3
"""
GWApproximation line_profiler 기반 프로파일링 스크립트.

사용법:
  1. input.ini가 있는 디렉토리에서 실행하거나, 아래 INPUT_DIR을 설정
  2. python profile_gw.py

결과:
  - 콘솔에 각 핵심 함수의 라인별 시간 출력
  - profile_gw_results.txt 에 결과 저장
"""
import sys
import os
import time
import io

# ---- 설정 ----
# input.ini가 있는 디렉토리 (None이면 현재 디렉토리)
INPUT_DIR = None
# 프로파일링할 SCF 반복 횟수 (빠른 테스트를 위해 줄임)
MAX_ITER = 3
# ---- 설정 끝 ----

if INPUT_DIR:
    os.chdir(INPUT_DIR)

# QAssemble src를 path에 추가
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from line_profiler import LineProfiler

# --- 프로파일 대상 함수들을 import ---
from QAssemble.FLatDyn import GreenBare, GreenInt, SigmaGWC, FLatDyn
from QAssemble.BLatDyn import PolLat, WLat, BLatDyn
from QAssemble.utility.Dyson import Dyson
from QAssemble.CorrelationFunction import CorrelationFunction

# --- LineProfiler 설정 ---
lp = LineProfiler()

# 핵심 함수들 등록
# 1) GW SCF 루프 전체
lp.add_function(CorrelationFunction.GWApproximation)

# 2) 각 클래스의 __init__ (객체 생성 시 계산 수행)
lp.add_function(GreenBare.__init__)
lp.add_function(GreenBare.Cal)
lp.add_function(GreenInt.__init__)
lp.add_function(GreenInt.CalMu0)
lp.add_function(GreenInt.SearchMu)
lp.add_function(GreenInt.NumOfE)
lp.add_function(GreenInt.UpdateMu)
lp.add_function(GreenInt.Occ)
lp.add_function(PolLat.__init__)
lp.add_function(PolLat.Cal)
lp.add_function(WLat.__init__)
lp.add_function(WLat.Cal)
lp.add_function(SigmaGWC.__init__)
lp.add_function(SigmaGWC.Cal)

# 3) Fourier transform 함수들 (Python 루프가 많은 부분)
lp.add_function(FLatDyn.T2F)
lp.add_function(FLatDyn.F2T)
lp.add_function(FLatDyn.K2R)
lp.add_function(FLatDyn.R2K)
lp.add_function(BLatDyn.T2F)
lp.add_function(BLatDyn.F2T)
lp.add_function(BLatDyn.K2R)
lp.add_function(BLatDyn.R2K)
lp.add_function(BLatDyn.RT2mRmT)
lp.add_function(BLatDyn.TauF2TauB)

# 4) Dyson solver
lp.add_function(Dyson.FLatDyn)
lp.add_function(Dyson.FLatStc)
lp.add_function(Dyson.FLocStc)
lp.add_function(Dyson.BLatDyn)
lp.add_function(Dyson.BLatStc)
lp.add_function(Dyson.BLocStc)


def run_gw():
    """input.ini를 읽고 GWApproximation을 실행"""
    from QuantumAssemble import Run
    runner = Run()
    # runner.RunDiagE()


def run_gw_direct():
    """
    input.ini 없이 README 예제와 동일한 파라미터로 직접 실행.
    input.ini가 없는 경우 사용.
    """
    from QuantumAssemble import InputReader, Run

    # README 예제 기반 최소 input
    crystal_dict = {
        'RVec': [[1,0,0],[0.5,0.866,0],[0,0,1]],
        'SOC': False,
        'CorF': 'F',
        'Basis': [[[0.33333,0.33333,0],1],
                  [[0.66667,0.66667,0],1]],
        'NSpin': 1,
        'NElec': 2,
        'KGrid': [6, 6, 1]  # 프로파일링을 위해 작은 그리드
    }

    hamiltonian_dict = {
        'OneBody': {
            'Hopping': {
                ((0,0),(1,0)): {
                    1.0: [[0,0,0],[-1,0,0],[0,-1,0]],
                },
            },
            'Onsite': {
                0: {(0,0): 0.0, (1,0): 0.0}
            }
        },
        'TwoBody': {
            'Local': {
                'Parameter': 'SlaterKanamori',
                'option': {
                    (0,(0,)): {'l': 0, 'U': 2.0, 'Up': 0.0},
                    (1,(0,)): {'l': 0, 'U': 2.0, 'Up': 0.0}
                }
            },
            'NonLocal': {
                ((0,0),(1,0)): {
                    0.20: [[0,0,0],[-1,0,0],[0,-1,0]],
                },
            }
        }
    }

    control_dict = {
        'Method': 'gw',
        'Prefix': 'profile_test',
        'NSCF': MAX_ITER,
        'Mix': 0.1,
        'T': 2000,
        'MatsubaraCutOff': 100,
        'ConstantW': 1.0
    }

    # input.ini 파일 작성
    with open('_profile_input.ini', 'w') as f:
        f.write(f"Crystal = {repr(crystal_dict)}\n\n")
        f.write(f"Hamiltonian = {repr(hamiltonian_dict)}\n\n")
        f.write(f"Control = {repr(control_dict)}\n")

    print("프로파일링용 input 파일 생성 완료: _profile_input.ini")
    print("이 파일로 실행하려면 input.ini로 복사 후 python profile_gw.py를 다시 실행하세요.")


if __name__ == "__main__":
    # input.ini 존재 여부 확인
    if not os.path.exists("input.ini"):
        print("=" * 60)
        print("input.ini가 없습니다.")
        print("=" * 60)
        print()
        print("방법 1: 기존 input.ini가 있는 디렉토리에서 실행")
        print("  cd /path/to/your/calculation && python /path/to/profile_gw.py")
        print()
        print("방법 2: 예제 input.ini 생성")
        print("  이 스크립트의 run_gw_direct()를 수정하여 사용")
        print()
        run_gw_direct()
        sys.exit(0)

    print("=" * 60)
    print("GWApproximation Line Profiling 시작")
    print("=" * 60)

    t_start = time.perf_counter()

    # line_profiler로 감싸서 실행
    lp_wrapper = lp(run_gw)
    lp_wrapper()

    t_total = time.perf_counter() - t_start
    print(f"\n전체 실행 시간: {t_total:.2f}s")

    # 결과 출력
    print("\n" + "=" * 60)
    print("Line Profiling 결과")
    print("=" * 60)
    lp.print_stats()

    # 파일로도 저장
    output = io.StringIO()
    lp.print_stats(stream=output)
    with open("profile_gw_results.txt", "w") as f:
        f.write(f"Total execution time: {t_total:.2f}s\n\n")
        f.write(output.getvalue())

    print(f"\n결과가 profile_gw_results.txt 에 저장되었습니다.")
