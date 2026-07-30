# EDMFT 수렴 실패 원인과 해결 기록

브랜치: `DLR_GW_EDMFT` · 기간: 2026-07-14 ~ 2026-07-27

프로덕션 EDMFT 실행에서 iteration이 진행되어도 수렴하지 못하고 오히려
발산하거나(μ 폭주), 물리적으로 말이 안 되는 impurity 입력이 CTQMC로
들어가던 문제들을 순서대로 추적해 고친 기록이다. 원인은 하나가 아니라
**서로를 가리고 있던 4개의 독립적인 결함**(1·3·4·5)이었고, 앞의 것을
고쳐야 뒤의 것이 드러나는 구조였다.

2번 절은 원인이 아니다 — 수사 도중 HF 장부를 오진해 만들었다가 되돌린
회귀이며, 같은 함정을 다시 밟지 않기 위해 기록으로만 남긴다.

---

## 1. Mixing이 통째로 no-op였다 — 감쇠가 전혀 걸리지 않음

**증상.** `Mix`, `MixingMethod`, `NPulay`를 어떻게 바꿔도 iteration 거동이
동일했다. 노이즈가 큰 QMC 샘플 하나가 100% 그대로 다음 iteration으로
전파되어 `Sigma_H = C*occ`와 μ 탐색까지 밀고 들어갔다.

**원인.** `CTQMC.PostProcessing`이 `sighimp/sigfimp/sigimp/pimp`의
`.Mixing()`을 호출하기 전에 `ctqmc/impurity_<iter>_<key>`로 `chdir` 하는데,
`hdf5file`은 `CorrelationFunction`에서 만들어진 **상대 경로 `'glob.h5'`**
였다. 따라서 `h5py.File(..., 'a')`가 매 iteration마다 그 스크래치
디렉터리 안에 **새 빈 파일**을 만들었고, `IO.Group`의 `require_group`이
트리를 새로 세우면서 `last`가 존재한 적이 없었다. `MixComponent`는 매번
"reset" 분기를 타고 입력을 그대로 돌려줬다. `Save()`는 `chdir` 복귀 후에
실행되므로 최상위 `glob.h5`에는 모든 데이터셋이 있지만 `Mixing` 그룹만
없었다 — 그래서 겉보기로는 정상이었다.

**해결** (`2830105`).
- `CorrelationFunction.hdf5path` 프로퍼티를 도입해
  `control['run']['fn'] + '.h5'`를 `os.path.abspath`로 **한 번** 해석하고,
  9개 생성 지점이 전부 이것을 쓰도록 했다. 해석은 lazy로 두어
  `object.__new__`로 만들어지는 테스트 인스턴스도 절대경로를 받는다.
  `control['run']['fn']` 자체는 HDF5 input 그룹에 스냅샷되어 재시작 시
  비교되므로 건드리지 않았다.
- `HDF5.MixComponent`: `iter > 1`인데 파일이 없으면 **예외**를 던지도록
  바꿔 이 부류의 실패가 다시는 조용히 지나가지 못하게 했다
  (`iter == 1`의 파일 생성은 정상). 어느 분기를 탔는지
  `num_history`/`npulay`와 함께 로깅하고, `iter > 1`의 reset은 경고다.

호출 순서는 의도적으로 그대로 뒀다 — `SigFImp.Cal`이 생성자에서
`sigh.s`를 빼기 때문에 `sighimp.Mixing()`이 `SigFImp` 생성보다 먼저
돌아야 한다. `chdir` 복귀 뒤로 옮겼다면 이 관계가 조용히 뒤집혔을 것이다.

---

## 2. (원인 아님) HF-in-level 시도와 그 되돌리기

> 이 항목은 **수렴 실패의 원인이 아니다.** 자체적으로 만들었다가 되돌린
> 회귀이므로, 기록 보존을 위해 남길 뿐 원인 목록에서는 제외한다.

**경위.** Δ(iω)가 큰 |ω|에서 감쇠하지 않는 것이 관측되자, static HF 장부를
의심해 `7ca8faa`/`de84bae`에서 lattice HF self-energy를 impurity level로
옮겼다 (`EImp.e = P(H0 − μ + Σ_H + Σ_F)`).

**왜 원인이 아니었나.** 같은 증상이 **이 커밋들 이전에도, 즉 `EImp`가 bare
level이고 `Hyb`가 HF를 한 벌만 빼던 시점에도 똑같이 존재했다.** HF 장부는
증상과 무관했다. 실제 원인은 다른 데 있었다 — mixing이 no-op이라 QMC
노이즈가 무감쇠로 전파됐고(1번), 그 노이즈가 5점 tail fit을 파탄시켜
causal projection이 매 iteration 실패하면서 비인과 데이터가 그대로 하류로
내려갔다(3번).

**시도가 만든 회귀.** `Hyb.Cal`은 여전히 같은 `sighloc`/`sigfloc`을 빼고
있었으므로 static HF가 장부에서 **두 벌** 차감됐고, Δ는 iteration 2부터
`-(Σ_H + Σ_F)` 크기의 offset을 **추가로** 얻었다. 원래 증상을 고치지 못한
채 새 문제만 얹은 셈이다.

**되돌리기** (`dd38129`). 원칙은 **static HF는 전체 장부에서 정확히 한 벌만
차감되어야 한다**로 정리했다. level에 HF를 넣으려면 impurity 자신의 HF(DC)를
반드시 같이 빼야 하는데 — CTQMC가 U로부터 자기 HF를 내부 생성하기 때문 —
순수 EDMFT에서는 lattice HF = impurity HF라 상쇄되어 bare level과 동치가
된다. 즉 EDMFT 경로에서 HF-in-level은 **애초에 아무것도 바꾸지 않는**
변경이었다. EDMFT 경로를 bare level로 되돌리고 `Hyb`에서
`sighloc + sigfloc + sigcloc`을 한 벌만 빼는 원래 상태로 복귀시켰다.
레퍼런스 관례(Ryee의 `edmft.py`, FullGWEDMFT의 `Eimpcomdc`가 EDMFT 극한에서
같은 것으로 환원)와 일치한다.

**교훈.** 증상이 나타난 시점과 코드 변경 시점을 먼저 대조했어야 했다.
증상이 변경 이전부터 있었다면 그 변경은 원인이 아니다.

**GWEDMFT 경로에는 회귀가 남아 있다** (`CorrelationFunction.py` ~line 1005):
`EImp`가 `hf_result.sigh.k`/`sigf.k`를 받는 동시에 `Hyb`가 같은 양의
Projection을 또 뺀다. EDMFT 경로만 되돌렸기 때문이다. FullGWEDMFT식 DC
처리(`hf_loc_result` 쪽 impurity HF를 `EImp`에 전달 + Hyb static 차감을
impurity HF 기준으로 정리)가 필요하다.

---

## 3. Causal projection이 거의 매 iteration 실패 — 원인은 tail fit

**증상.** 프로덕션 실행에서 fermion `Hyb`가 약 109회 QP infeasible,
boson `BWeiss`가 약 86회 tail-decay gate에 걸려 실패했고, 그때마다
**가공되지 않은 비인과적 Δ/cf가 그대로 CTQMC로 전달**됐다.

**원인.** tail moment `c1..c3`를 signed-uniform 그리드의 **연속된 5점**에서
피팅했다. condition number가 ~5e11이라 `c2`/`c3`가 쓰레기값이었고, 그
쓰레기값을 QP의 **hard equality**로 강제하니 feasible set이 비었다.

**해결** (`7e1e571`).
- `Fourier`: `_tail_fit_indices`가 최고 |ω| 한 decade에 걸쳐 **log-spaced
  ~24점**을 고르도록 했다(uniform 그리드 대상).
  `FermionTailCoefficients`/`BosonTailCoefficients`에
  `log_spaced`/`return_sigma`(lstsq 1σ 불확도)를 추가했다. 기본값은 legacy
  피팅과 bit-identical하게 유지.
- `Causal`: `CausalProjection.project`에 `equality_penalty`를 추가해
  slack pair(`A x - b = s⁺ - s⁻`)로 **exact-L1 elastic 모드**를 만들었다 —
  feasible set이 절대 비지 않는다. hard equality를 먼저 시도하고 실패 시에만
  elastic으로 재시도한다(뻣뻣한 μ~1e8을 피하기 위함). fermion/boson
  projector가 `moment_sigma`를 전달하고 `μ = 1/σ`를 node-residual +
  `auto_floor`로 바닥 처리한다.
- Bosonic **non-decaying offset**(CTQMC 노이즈가 Dyson을 통해 들어온 것)은
  tail gate를 올리는 대신 분리·경고 후 출력에서 **버린다**.
- 솔버가 완전히 죽는 경우엔 regularized fit(`rcond=1e-2`) + sign-clip
  fallback을 반환한다 — 원본 비인과 데이터가 하류로 내려가는 경로를 없앴다.
- `BWeiss.Cal`에 `Hyb.Cal`과 같은 안전망을 붙였다(이전엔 없어서 솔버 크래시
  하나가 실행 전체를 죽였다).
- 검증 도구 `validate_glob_causal.py`: 저장된
  `hyb.<it>.<key>`/`bweiss.<it>.<key>_correlated_uniform`을 read-only로
  재생해 success/skip/elastic/clip/offset 통계를 낸다.

---

## 4. Fallback이 dyn.json을 0으로 채움 — retarded interaction 소실

**증상.** 3번을 고쳐 fallback이 항상 동작하게 만들자, 이번엔 QP가 실패한
iteration에서 CTQMC의 `dyn.json` retarded interaction이 **전부 0**으로
들어가는 것이 확인됐다. clipped fallback이 부호가 틀린 계수를 전부 0으로
깎아버린 결과였다. 상호작용이 사라진 impurity 문제를 푸니 수렴할 리가 없다.

**해결** (`2d70e3a`, FullGWEDMFT의 `causal_boson`/`pimpbrd`/`ximpbrd`/
`wlocbrd`/`f0brd` wiring 이식).
- `ProjectBosonComponentWithFallback`의 실패 우선순위를 재구성:
  **elastic QP → 이전 iteration의 투영 결과(`fallback_channel`) → clipped**.
  clipped 결과가 degenerate하면(`|clipped| < 1e-3·|target|`) 0 대신
  **비투영 채널을 경고와 함께 반환**한다(zero-guard). 0은 이제 `dyn.json`에
  도달하지 못한다.
- `*_brd_prev` HDF5 캐시를 BWeiss/Chi/PImp/WLoc에 추가
  (`IO.Read/WriteProjectionCache`, `IO.OverwriteMixingLast`) — QP 실패 시
  이전 iteration의 **투영된** 결과로 되돌아갈 수 있게 했다.
- `BWeiss`의 `p is None` 게이트 제거 — bare/first-iteration bath도 투영한다.
- `WLoc.Cal`에 causal projection 추가(`wlocbrd` 대응: `cf = f - v`를
  zero-static 정책·`grid="dlr"`로 투영 후 정확한 `v`를 재가산).
- `PImp`: zero-static 정책(`pimpbrd static_mode="zero"`) 채택, mixing 후
  **재투영**, mixing history의 `last`를 투영값으로 덮어쓰기(투영값이 다음
  fold가 되는 FullGWEDMFT 의미론), stale `f_uniform` 수정.
- 첫 투영 시 설치된 `qpsolvers` 백엔드를 한 번 로깅한다(실행 환경에 솔버가
  없어서 실패하는 경우를 즉시 식별하기 위함).
- `validate_glob_causal.py`에 pimp/wloc 검증(DLR 그리드 감지, static-fit
  모드)과 `--run-dir`로 `dyn.*.json`의 all-zero 채널 스캔을 추가했다.

테스트: 241 passed (신규 20, 계약 변경으로 의도적 재작성 3).

---

## 5. CTQMC 입력 자체에는 감쇠가 없었다

**증상.** 1번을 고쳐 self-energy mixing이 실제로 돌기 시작한 뒤에도,
CTQMC가 매 iteration 받는 **입력**(hybridization, retarded interaction)은
직전 iteration 값에서 통째로 갈아치워졌다. 상류의 self-energy만 damping해서는
solver가 보는 bath의 iteration 간 진동이 잡히지 않았다.

**해결** (`a5508c8`). FullGWEDMFT의 `deltamix`/`Uweisstot_mix`에 대응하는
입력 mixing을 추가했다.
- `FWeiss.Mixing(iter, control)` (`FLocDyn.py`) — component `"hyb"`,
  hyb를 혼합 → fermion causal 재투영(try/except; fermion 투영엔
  `fallback_matrix`가 없다) → `OverwriteMixingLast` → `Cal()` 재실행으로
  `hyb.json` 소스 `h` 갱신.
- `BWeiss.Mixing(control)` (`BLocDyn.py`) — component `"bweiss"`, `cf`를
  혼합 → boson 재투영(`_brd_prev` fallback) → `last` 덮어쓰기 →
  `f`/`f_uniform`/`cf_uniform`/`t`/`ct` 전부 재산출. PImp의 `utilde`·WImp와
  `dyn.json`의 일관성을 유지하기 위함이며, 이는 FullGWEDMFT의 mixed-U
  의미론과 같다.
- `CTQMC.PreProcessing`에서 `hyb.json`/`dyn.json` 작성 직전에 무조건 호출
  (iteration 1은 pass-through).

별도 플래그 없이 무조건 혼합하기로 했고, 파라미터는 `sig`/`pimp`가 쓰는
것을 따른다 — 현재는 단일 `control["mix"]`이라 둘 다 같은 값이다. 나중에
`sig`/`pimp` mix 노브가 분리되면 각각 매핑할 의도. 상류 self-energy 혼합과
겹쳐 **이중 감쇠**가 되는 점은 인지하고 선택한 것이다.

---

## 6. `PImp` 부호 규약이 반대였다 — 3·4번 증상의 공통 원인

**증상.** 3번에서 "causal projection이 거의 매 iteration 실패"로, 4번에서
"fallback이 dyn.json을 0으로" 로 기록한 증상. 당시엔 tail fit만 원인으로
진단했으나, `PImp` 자체가 부호가 반대여서 투영이 **구조적으로 불가능**했다.

**원인.** `BLocDyn.py`의 `PImp.Cal`이 `Dyson(chi, -utilde)` = `χ(1+Uχ)⁻¹`,
즉 **양수**를 만들고 있었다. 참조 구현은 둘 다 음수 규약이다:

- `FullGWEDMFT/src/ComDMFT/Src/qft_comlocal.F:406-413` → `P = -(1-χU)⁻¹χ`
- `gw_edmft_v208/edmft.py:2676` → `P = -χ(1-Uχ)⁻¹`

부호 오류가 **두 개**였다 (`sigma` 인자 부호 + 바깥 마이너스 누락). 하나만
고치면 맞지 않는다 — 실측: 바깥만 1.0, 안쪽만 2.3, 둘 다 5.6e-17.

**왜 드러나지 않았는가.** 실제 `CausalBosonProjector`로 측정한 결과, 양수
입력에 대해 QP 솔버가 **전부 실패**하고 clipped fallback이 발동해 값을 강제로
음수화했다: `+0.200 → −0.478` (부호 반전 **및** 2.4배 과대, 상대오차 339%).
즉 투영이 부호를 조용히 덮으면서 크기를 망가뜨리고 있었다. `coefficient_sign=-1`
(`x ≤ 0` 요구)에 양수를 밀어넣던 상태이므로 실패는 필연이었다.

**해결.** `BLocDyn.py:1119` 한 줄:

```python
self.f_uniform = -self.Dyson(self.chi_boson_uniform, utilde)
```

`Dyson._solve_bosonic`은 우측 곱 형태 `g0(1-σg0)⁻¹`를 반환하며, push-through
항등식으로 `(1-g0σ)⁻¹g0`와 같다. 따라서 `-Dyson(χ,U) = -(1-χU)⁻¹χ`로 참조식과
정확히 일치하고, **비가환 다체 궤도에서도 피연산자 재배열이 필요 없다**.
`(1-Uχ)⁻¹χ`는 다른 행렬이므로(오차 2.0) 이 점이 핵심이다 — 기존 헬퍼가
문서화되지 않은 채 마침 올바른 쪽을 주고 있었다. `Dyson.py`에 이 공식을
docstring으로 명시해 재발을 막았다.

**건드리지 않은 것 (검증 완료).** 이들은 이미 음수 규약을 전제하므로 자동으로
옳아진다 — 함께 뒤집으면 정상 동작 중인 코드가 깨진다:

- `WImp.Cal` `Dyson(utilde, polarization)` — `P`를 그대로 받음
- `BWeiss.Cal` `Dyson(w, -p)` = `(W⁻¹+p)⁻¹` — FullGWEDMFT `Uweiss⁻¹ = Wloc⁻¹ + Pimp`와 이미 일치 (행렬 케이스 1.4e-15 확인)
- `PolC.__call__` `+pimp` — GW 자체 `P`도 음수
- `W.Cal`, `Convergence`(절대차 → 부호 무관), `validate_glob_causal.py`(이미 `coefficient_sign=-1`)

**검증.**

- QP 경고 **0건** (수정 전: 전 솔버 실패), 편차 339% → 2.2%
- `PImp = -0.750000` — 참조 `-χ/(1-Uχ)`와 일치
- 격자 `W/v = 0.4000` (차폐). 수정 전 `1.60`으로 **반차폐**였다
- 회귀 테스트 3건 신설 (`test_pimp_sign_convention.py`): 부호, 비가환 블록에서의
  참조식 등가성(잘못된 순서와 구별되는지도 함께 검증), `P = U⁻¹ - W⁻¹` 상호일관성.
  세 테스트 모두 수정 전 코드에서 **실패함을 확인**했다.
- 테스트: 258 passed (기존 실패 2건 유지, 신규 실패 0)

**⚠️ 재시작 주의.** `glob.h5`에 이전 **양수** 값이 남는다:
`<group>/PImp/pimp_brd_prev.<key>`, `<group>/Mixing/<key>/pimp/{last,input_history,residual_history}`.
`ReadProjectionCache`는 shape만 검증하고 부호는 보지 않으며, stale fallback은
`Causal.py`에서 **그대로 복사되어 반환**되므로 부호가 반전된 값이 조용히
하류로 흐른다. 이 수정 이후 기존 `glob.h5`에서 재시작하지 말 것. 필요하면 위
데이터셋을 삭제하거나 처음부터 재실행. 신규 실행은 `iter==1`에서 mixing 이력이
자동 리셋되므로 안전하다.

**⚠️ 결과가 바뀐다.** 수정 전에는 격자 분극의 불순물 항 부호가 반대였으므로
`W`/`Uweiss`/`Σ`/`μ` 모두 다른 고정점에 수렴해 있었다. 이전 결과와의 정량적
차이는 예상된 것이며 회귀가 아니다. 또한 분모가 `1+Uχ`(무조건 안정화)에서
`1-Uχ`로 바뀌므로, 전하 불안정 근처에서 이전에 "수렴"했던 계산이 이제 정당하게
발산할 수 있다.

## 전체 그림

실제 수렴 실패 원인은 **4개**였다 (2번은 원인이 아니라 자체 회귀).

| # | 문제 | 수렴에 미친 영향 | 커밋 |
|---|------|------------------|------|
| 1 | 상대경로 `glob.h5` → Mixing no-op | 감쇠 0, QMC 노이즈 100% 전파 | `2830105` |
| 3 | 5점 tail fit → QP infeasible | 투영 실패 시 비인과 데이터 통과 | `7e1e571` |
| 4 | clipped fallback → dyn.json 전부 0 | 상호작용이 사라진 impurity 문제 | `2d70e3a` |
| 5 | CTQMC 입력에 damping 없음 | bath의 iteration 간 진동 미제어 | `a5508c8` |
| 6 | `PImp` 부호 반대 → QP 구조적 실패 | 3·4의 공통 원인, 격자 `W`가 반차폐 | (본 작업) |
| ~~2~~ | HF-in-level → 이중 차감 | **원인 아님** (자체 회귀) | `dd38129`로 되돌림 |

인과관계로 보면 1이 3·4를 증폭시키고 있었다 — mixing이 죽어 있으니 QMC
노이즈가 그대로 tail로 들어가 QP를 깨뜨렸고, QP가 깨지니 fallback이 발동해
비인과 데이터나 0이 하류로 흘렀다. 그래서 1을 고치기 전에는 3·4의 수정
효과를 측정할 수 없었다.

다만 3·4의 원인을 tail fit만으로 본 것은 **불완전한 진단이었다**. 6번에서
드러난 대로 `PImp`는 부호가 반대여서 tail이 아무리 깨끔해도 QP가 성립할 수
없었다. 즉 3·4의 실패에는 노이즈성 원인(1·3)과 구조적 원인(6)이 겹쳐 있었고,
6을 고치기 전까지는 `PImp` 채널의 투영이 100% fallback으로 처리되고 있었다. 2번(HF 장부)을 의심한 것은 오진이었고, 증상이
그 커밋들 이전부터 있었다는 사실이 이를 확인해 준다.

## 남은 과제

- **GWEDMFT 경로의 HF 이중 차감** — EDMFT 경로에서는 되돌렸으나 여기는 살아 있음.
- **`eimp.Projection` stale reference** (`CorrelationFunction.py` ~599-602).
- **`FLatDyn`/`BLatDyn`의 legacy 5점 tail fit** — 3번의 수정이 아직 적용되지 않음.
- **μ 폭주의 근본 원인** — under-relaxation이 필요하다는 진단까지만 되어 있음.
- **`zero_ratio = 1e-3` 임계값 보정** — 실제 실패 클러스터 실행 데이터로
  캘리브레이션 필요.
- **실전 검증**: 5번(입력 mixing) 이후의 `glob.h5` 수렴 거동은 아직 확인하지 않았다.
- **6번의 실전 검증**: 부호 수정 후 실제 EDMFT 실행에서 `PImp` QP 경고가
  사라지는지, 수렴이 개선되는지는 아직 확인하지 않았다.
- **`eimp.Projection` stale reference** (`CorrelationFunction.py:628,630`) —
  `EImp` 생성(638행)보다 위에서 `eimp.Projection`을 호출한다. 함수 스코프가
  루프 반복 간 유지되므로 2회차부터는 **이전 반복의** `EImp` 객체를 쓴다.
  현재는 무해하다 — `Projection`이 `self.projector`(런 불변)와 인수 `key`만
  읽고 `self.e`/`self.mu`는 쓰지 않기 때문. 다만 `Projection`이 인스턴스
  상태에 의존하게 바뀌면 조용히 stale 값을 쓰게 되므로, `EImp` 생성 블록을
  위로 올리거나 631행처럼 `gloc.Projection`으로 통일하는 것이 안전하다.
  (GWEDMFT 경로는 이미 생성이 먼저다.)
- **K0 차폐 shift 전면 누락** — 참조의 `shift = -2K0_old·n + 2K0_new·n`에
  해당하는 로직이 코드 전체에 없다 (`grep K0` 결과 0건). ComCTQMC가 내부적으로
  더하는 것은 정적 sector 에너지(`0.5·Q·D0·Q`)이지 이 차분 보정이 아니다.

검증 절차:

```bash
PYTHONPATH=$PWD/src ~/.pyenv/versions/3.13.9/bin/python -m pytest src/QAssemble/utility/test/ -q
```

(기존 실패 2건 — `test_method_impurity_action.py`의 `test_hfloc_builds_...`/
`test_gwloc_builds_...` — 는 클린 HEAD에서도 실패하는 것으로 본 작업과 무관하다.)
