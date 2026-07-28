# DLR→uniform Matsubara 변환 수정 보고서

**대상 브랜치:** `DLR_GW_EDMFT`
**커밋 범위:** `d135cf1` … `83bb232` (4개)
**작성일:** 2026-07-28

---

## 1. 문제

`ctqmc/impurity_32_1/hyb.json`의 hybridization 실수부가 고주파에서 0으로 수렴하지
않았다.

| 항목 | 관측 | 정확값 |
|---|---|---|
| `Im Δ·ω` | -6.04로 수렴 | -6.04 (정상) |
| `Re Δ` (ω≈187) | 부호 반전, +2.4e-4까지 상승 | 음수 |
| `Re Δ` (ω=460) | +1.3e-4 잔류 | `-7.64/ω² = -3.6e-5` |

---

## 2. 원인 규명

### 2.1 입력 데이터는 정상이었다

`glob.h5`에 저장된 `GLoc`, `EImp`, `SigCImp`, `SigHImp`, `SigFImp`로 Dyson 식을
직접 재구성해 DLR 노드에서의 raw Δ를 얻었다.

```
Re Δ·ω²  =  -7.43  -7.38  -7.52  -7.55  -7.58  -7.62  -7.74  -7.64  -7.77
            (ω = 33 … 464, 정확값 -7.64인 상수)
```

노드는 깨끗하다. **오염은 상류가 아니다.**

### 2.2 변환 자체가 원인

그 *깨끗한 배열*을 `MatsubaraDLR2UniformGrid`에 그대로 넣었을 때:

```
Re Δ·ω²  =  -1.2   +344   +3538   -1328   -895   +6280   +10475
            (ω = 50 … 460)
```

3자릿수 오차가 **변환 단계에서만** 주입된다.

기전은 노드→계수 LU 해에서의 catastrophic cancellation이다. 계수 노름이
`sum|c| = 6.0e+14`까지 폭발하고, 상쇄 후 남은 반올림 잡음이 결과를 지배한다.
그 결과 변환은 **자기 입력 노드조차 재현하지 못한다** — 실데이터 기준 노드 오차
3.12, ω=138.89에서 실제 값 2.8e-4가 부호까지 뒤집힌다.

> **주의할 지표:** `cond(dlrmf2cf) = 8.91e+05`는 낮아 보이지만 이는 재스케일된
> 다른 계이며 정확도를 지배하지 않는다. 이 숫자로 LU 경로가 건전하다고 결론
> 내리면 안 된다. DLR로 정확히 표현 가능한 입력에서 계수 복원 상대오차는 **0.999**
> (복원 `sum|c|` 19725 vs 참값 75)였다.

### 2.3 하류 전파 경로

```
Hyb.Cal  →  UniformGrid()  →  CausalProjection(grid="uniform")
             (깨진 변환)         └─ tail fit 창 = 상위 1 decade
```

`Fourier._tail_fit_indices`가 고르는 창을 측정하니 **[46.46, 464.61]** — 손상
구간 전체와 정확히 겹친다. 여기서 fit된 상수항 `c0`가 0 대신 ~+1.2e-4로 나오고,
projector가 이를 결과에 상수 offset으로 재삽입한다. 해당 구간 `Re Δ`는 3.6e-5라
압도당하고, `Im Δ`는 1e-2라 상대적으로 살아남았다.

---

## 3. 수정 내용

### 3.1 방식

DLR 계수를 **전혀 만들지 않고** 노드값에서 uniform grid로 직접 보간한다.

1. **Hermitian folding** — `f(-ω) = conj(f(ω))`로 음수 노드를 양수축에 반사
2. **변수 변환** — `u = 1/ω` (tail이 이 변수에서 저차 다항식)
3. **Re/Im 독립 보간** — 실수부·허수부를 분리해 각각 cubic spline

### 3.2 folding이 필수인 이유

양수 노드만 쓰는 것은 틀린다. `MatsubaraFermionUniform`이 grid 크기를 **부호 있는**
극값에서 정하기 때문에([DLR.py:50](../src/QAssemble/utility/DLR.py#L50)), uniform grid가
최대 양수 노드를 넘어갈 수 있다.

| beta/cutoff | 최대 양수 노드 | uniform 최대 | 외삽 |
|---|---|---|---|
| 100/300 (프로덕션) | 464.610 | 464.610 | 없음 |
| **20/8 (테스트 fixture)** | 6.440 | 11.153 | **73% 초과** |
| 10/10 | 11.624 | 19.164 | 발생 |
| 200/400 | 300.886 | 564.214 | 발생 |

**5개 중 4개가 외삽**하며 프로덕션이 예외다. `np.interp`는 clamp하므로 양수 전용
구현은 조용히 평평한 직선을 반환한다. folding은 관계식이 오차 0.00e+00으로
정확히 성립하며, 커버리지를 복원하고 사용 가능 노드를 47→81로 늘린다.

folding은 **래퍼에서** 수행한다. 행렬 값 관계는 `G(-iω) = G(iω)†`(켤레 + 전치)인데
평탄화된 코어에는 궤도축이 없어 전치가 불가능하기 때문이다.

### 3.3 API

```python
MatsubaraDLR2UniformGrid(ff, sign=-1, method="interp")   # 기본값
MatsubaraDLR2Uniform(ff, sign=-1, method="dlr")          # 기존 경로 유지
```

`method="dlr"`을 남긴 이유는 **잘 조건화된 설정에서는 기존 경로가 여전히
기계정밀도로 정확**하기 때문이다(테스트 fixture cond 2.5e+12 vs 프로덕션 3.0e+17).
삭제가 아니라 보존이 옳은 판단이었다.

### 3.4 변경하지 않은 것

`Fourier._tail_fit_indices`는 **의도적으로 손대지 않았다.** 창이 잘못 고른 게
아니라 손상된 값을 읽고 있었을 뿐이다. 변환이 고쳐지면 같은 창이 정상 데이터를
fit한다.

---

## 4. 검증 결과

### 4.1 실데이터 (iterations 27–32, tail window [46, 231])

| iter | 기존 `Re Δ·ω²` 범위 | 현재 `Re Δ·ω²` 범위 | 노드오차 기존/현재 |
|---|---|---|---|
| 27 | [-485.1, +330.3] | [-9.17, +0.81] | 7.6e-02 / 2e-16 |
| 28 | [-426.6, +302.4] | [-8.46, +3.54] | 7.5e-02 / 5e-16 |
| 29 | [-411.7, +278.2] | [-7.32, +0.98] | 7.9e-02 / 4e-16 |
| 30 | [-405.8, +292.4] | [-7.54, -0.18] | 7.9e-02 / 4e-16 |
| 31 | [-409.3, +288.2] | [-7.30, -0.43] | 7.9e-02 / 4e-16 |
| 32 | [-416.8, +284.4] | [-7.29, -1.34] | 7.9e-02 / 4e-16 |

정확값은 **-7.64 상수**. 현재 경로는 노드를 **기계정밀도**로 통과한다.

### 4.2 새로 드러난 사실 — `Im`도 오염되어 있었다

노드 재현 오차를 Re/Im으로 분리하면:

| | 기존 Re | 기존 Im | 현재 Re | 현재 Im |
|---|---|---|---|---|
| 최악값 (27–32) | 3.7e-02 | **7.6e-02** | 2.2e-16 | 4.4e-16 |

**`Im`의 절대 오차가 `Re`보다 2배 이상 컸다.** 원래 진단에서 "Im은 무사"로 본 것은
`Im Δ` 자체가 1e-2로 커서 상대적으로 가려졌기 때문이다. 초기 판단을 이 지점에서
정정한다.

### 4.3 테스트

```
255 passed, 2 failed
```

실패 2건은 `test_method_impurity_action.py`의 기존 실패로 이번 변경과 무관하다
(baseline 247 passed + 동일한 2 failed). 신규 테스트 8건 추가.

보손 출력은 모든 `method` 값에서 **bit-identical**(`max diff: 0.0`)로 확인했다.

---

## 5. 커밋 구성

| 커밋 | 내용 |
|---|---|
| `d135cf1` | `_interp_to_grid`에 `variable`/`kind` 추가(기본값은 기존 동작 보존), `_fold_to_nonnegative` 헬퍼. 순수 추가 |
| `6b2e99f` | `method="interp"` 기본 전환 + 신규 테스트 8건 |
| `8df8b6e` | 진단 스크립트에 출시 경로 추가 |
| `83bb232` | Re/Im 비교 plot 스크립트 + 그림 2장 |

총 8 files, +1199 / -17.

---

## 6. 남은 작업

### 6.1 보손 경로 (우선순위 높음)

- **조건수 3.22e+18** — 페르미온(2.96e+17)보다 나쁘다
- 현재 bit-identical로 **미수정 상태**
- `ν=0`이 노드·uniform grid 양쪽에 실재해 `1/ω`가 발산 → 시프트 변수
  `1/(ω+ω₀)`, `ω₀=2π/beta` 필요
- 영향 경로: `BWeiss.Cal`([BLocDyn.py:1274](../src/QAssemble/BLocDyn.py#L1274)),
  `Chi.Cal`(:972), `PImp.Mixing`(:1151), `WImp.Cal`(:1398)
- **`dyn.json`이 CTQMC 입력이므로 solver 입력이 바뀐다** — 진단 필요

### 6.2 회귀 방어의 한계

신규 테스트는 합성 데이터(DLR로 정확히 표현 가능한 다중극)에서는 **기존 경로도
통과한다.** 실제 판별력은 `glob.h5`의 진짜 self-energy에서만 나온다(노드오차
3.12 vs 0.03). 실데이터 회귀를 막으려면
[docs/compare_dlr2uniform.py](compare_dlr2uniform.py)를 CI에 넣는 것이 확실하다.

### 6.3 검토 대상

- `constraint_tol="auto"` 분기([FLocDyn.py:1364](../src/QAssemble/FLocDyn.py#L1364),
  [BLocDyn.py:1279](../src/QAssemble/BLocDyn.py#L1279)) — 두 주석 모두 "보간 잡음이
  1e-8 기본값을 초과"를 근거로 든다. 정확도가 크게 개선됐으므로 이제 불필요하게
  느슨할 수 있다
- `eps=1e-15` 기본값([DLR.py:21](../src/QAssemble/utility/DLR.py#L21)) — rank 95와
  1e17 조건수를 만드는 원인. 이번 버그의 우회책이 아니라 **독립적 설계 결정**으로
  재검토할 가치가 있다

---

## 7. 산출물

| 파일 | 용도 |
|---|---|
| [compare_dlr2uniform.py](compare_dlr2uniform.py) | 재구성 방식별 점수화 (`--iters 27-32 --tail 46,231`) |
| [plot_hyb_dlr_vs_uniform.py](plot_hyb_dlr_vs_uniform.py) | Re/Im 비교 plot (`--with-sigma`) |
| [dlr2uniform_compare.png](dlr2uniform_compare.png) | 방식별 tail 비교 |
| [hyb_dlr_vs_uniform.png](hyb_dlr_vs_uniform.png) | hybridization 6 iteration × 4 패널 |
| [sigma_stored_vs_current.png](sigma_stored_vs_current.png) | 저장된 uniform vs 현재 변환 |

> `edmft/Hyb`에는 `_uniform` 데이터셋이 없어 hybridization의 기준은 DLR 노드다.
> 저장본과의 직접 비교가 가능한 곳은 `edmft/SigCImp/*_uniform`뿐이며, 이는 옛
> 코드가 같은 깨진 변환을 거쳐 실제로 기록한 데이터다.
