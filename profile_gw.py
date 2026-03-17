#!/usr/bin/env python3
"""
GWApproximation line_profiler 기반 프로파일링 스크립트.

사용법:
  1. input.ini가 있는 디렉토리에서 실행
  2. python profile_gw.py

결과:
  - 콘솔에 각 핵심 함수의 라인별 시간 출력
  - profile_gw_results.txt 에 결과 저장

주의: 실행 전 __pycache__ 삭제 권장
  find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null
"""
import sys
import os
import time
import io
import subprocess

# ---- 설정 ----
INPUT_DIR = None
# ---- 설정 끝 ----

if INPUT_DIR:
    os.chdir(INPUT_DIR)

# QAssemble src를 path에 추가
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

# __pycache__ 삭제 (소스 변경 시 stale .pyc 방지)
for root, dirs, files in os.walk(src_dir):
    for d in dirs:
        if d == '__pycache__':
            import shutil
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

from line_profiler import LineProfiler


def run_gw():
    """input.ini를 읽고 GWApproximation을 실행하며 line profiling 수행"""

    # --- import는 여기서 수행 (fresh import, __pycache__ 삭제 후) ---
    from QAssemble.FLatDyn import GreenBare, GreenInt, SigmaGWC, FLatDyn
    from QAssemble.BLatDyn import PolLat, WLat, BLatDyn
    from QAssemble.utility.Dyson import Dyson
    from QAssemble.CorrelationFunction import CorrelationFunction

    # --- LineProfiler 생성 및 함수 등록 (import 직후, 동일 객체 보장) ---
    lp = LineProfiler()

    # GW SCF 루프 전체
    lp.add_function(CorrelationFunction.GWApproximation)

    # 각 클래스의 핵심 메서드
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

    # Fourier transform
    lp.add_function(FLatDyn.T2F)
    lp.add_function(FLatDyn.F2T)
    lp.add_function(FLatDyn.K2R)
    lp.add_function(FLatDyn.R2K)
    lp.add_function(BLatDyn.T2F)
    lp.add_function(BLatDyn.F2T)
    lp.add_function(BLatDyn.K2R)
    lp.add_function(BLatDyn.R2K)
    lp.add_function(BLatDyn.RT2mRmTDLR)
    lp.add_function(BLatDyn.TauF2TauB)

    # Dyson solver
    lp.add_function(Dyson.FLatDyn)
    lp.add_function(Dyson.FLatStc)
    lp.add_function(Dyson.FLocStc)
    lp.add_function(Dyson.BLatDyn)
    lp.add_function(Dyson.BLatStc)
    lp.add_function(Dyson.BLocStc)

    # --- 실행: Run() 생성자가 내부적으로 GW 계산을 호출 ---
    from QuantumAssemble import Run

    @lp
    def _run():
        runner = Run()

    t_start = time.perf_counter()
    _run()
    t_total = time.perf_counter() - t_start

    print(f"\n전체 실행 시간: {t_total:.2f}s")

    # --- 결과 출력 ---
    print("\n" + "=" * 60)
    print("Line Profiling 결과")
    print("=" * 60)
    lp.print_stats()

    # 파일로 저장
    output = io.StringIO()
    lp.print_stats(stream=output)
    with open("profile_gw_results.txt", "w") as f:
        f.write(f"Total execution time: {t_total:.2f}s\n\n")
        f.write(output.getvalue())

    print(f"\n결과가 profile_gw_results.txt 에 저장되었습니다.")


if __name__ == "__main__":
    if not os.path.exists("input.ini"):
        print("=" * 60)
        print("input.ini가 없습니다.")
        print("=" * 60)
        print()
        print("input.ini가 있는 디렉토리에서 실행하세요:")
        print("  cd /path/to/your/calculation && python /path/to/profile_gw.py")
        sys.exit(0)

    print("=" * 60)
    print("GWApproximation Line Profiling 시작")
    print("=" * 60)

    run_gw()
