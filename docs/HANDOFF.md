# FDM 환경의 SMBH 병합 지연 모델 업무인수인계서

- 작성일: 2026-08-09 KST
- 원 프로젝트: `/home/kjhan/BACKUP/FDM`
- 권장 새 프로젝트명: `FDM_SMBH_DELAY`
- 원 코드 기준 commit: `7ba393fb9a1156313cfc20b488128fdf55ed1c1d`
- 원 코드 branch: `fdm-mg-convergence`
- 상태: 과학·수치 설계 완료, 독립 toy model은 아직 구현하지 않음

---

## 1. 한 페이지 요약

lagRamses의 SMBH sink 병합은 실제 black-hole coalescence가 아니다. 현재 구현은 두 sink가
`rmerge * dx_min` 안에 들어오고 결합되어 있으면 한 sink로 즉시 합친다. 이 수치 병합 반경은
matched-v4에서 약 1.73 kpc, 64/128 Mpc production에서 약 23.1 kpc이다. 실제 SMBH 병합은 그
뒤에도 아직 다음 세 구간을 거쳐야 한다.

1. `4 pixels -> 1 pc`: gas/star/FDM 환경을 포함한 unresolved pairing
2. `1 pc -> 0.01 pc`: soliton 안에서 FDM wave dynamical friction과 gravitational cooling
3. `< 0.01 pc`: gravitational-wave inspiral과 coalescence

새 프로젝트의 첫 목표는 **구간 2만** 독립적인 ODE toy model로 계산하는 것이다. 입력은 SMBH
질량, 위치, 속도, FDM 입자질량, soliton 질량/밀도 profile이다. 출력은
`1 pc -> 0.01 pc` 지연시간, 궤도·에너지·각운동량 이력, FDM으로 전달된 에너지와 근사식 유효성
flag이다.

이 프로젝트는 현재 FDM hybrid-method 논문의 완료 조건이 아니다. 검증되면 lagRamses의 각
수치적 sink merger에 후처리로 적용하여

```text
numerical sink-merger time -> inferred physical coalescence time
```

을 계산한다. 모델이 안정되기 전에는 lagRamses의 실제 sink dynamics나 현재 실행 중인 simulation을
변경하지 않는다.

---

## 2. 과학 질문과 산출물

### 핵심 질문

> FDM halo에서 lagRamses가 두 SMBH sink를 수치적으로 합친 시각과 두 black hole이 실제로
> coalesce하는 시각 사이에는 얼마의 지연이 있는가?

### 사건별 최종 산출물

- 수치 병합 시각과 redshift: `t_sink`, `z_sink`
- 물리적 병합 시각과 redshift의 추정 분포: `t_coal`, `z_coal`
- 총 지연시간:

  \[
  \Delta t_{\rm total}
  =\Delta t_{\rm kpc\rightarrow pc}
  +\Delta t_{\rm FDM}
  +\Delta t_{\rm GW}
  \]

- `z=0` 이전 coalescence 여부
- FDM/CDM 환경에서의 delay-time 및 merger-rate 차이
- 각 결과의 validity flag와 모델 불확실성

### 새 프로젝트의 첫 범위

첫 release는

\[
\Delta t_{\rm FDM}=t(0.01\,{\rm pc})-t(1\,{\rm pc})
\]

만 계산한다. kpc-to-pc와 GW 구간은 인터페이스만 정의하고 구현은 뒤로 미룬다.

---

## 3. 현재 lagRamses SMBH 병합 구현

### 실제 사용되는 source

빌드는 `PATCH=../patch/lagRamses`이지만 Makefile의 `VPATH`가 `patch/cuRamses`까지 검색하므로
현재 sink object는 다음 source에서 만들어진다.

- 병합 코드:
  [`sink_particle.kjhan.f90`](/home/kjhan/BACKUP/FDM/lagramses_science_clean/patch/cuRamses/sink_particle.kjhan.f90:1026)
- 실제 거리 조건:
  [`sink_particle.kjhan.f90`](/home/kjhan/BACKUP/FDM/lagramses_science_clean/patch/cuRamses/sink_particle.kjhan.f90:1170)
- 물리 단위:
  [`units.f90`](/home/kjhan/BACKUP/FDM/lagramses_science_clean/amr/units.f90:22)
- 128 Mpc production 설정:
  [`cosmo.nml`](/home/kjhan/BACKUP/FDM/clean_production/cosmo.nml:99)

### 현재 판정

코드는

\[
d \le r_{\rm merge}\,dx_{\min}
\]

인 sink들을 FOF group으로 묶는다. `vrel_merge=.true.`이면 추가로

\[
E_{\rm kin,COM}<E_{\rm grav}
\]

을 요구한다. 조건을 만족하면 두 개 이상의 sink를 즉시 하나의 질량중심 sink로 바꾼다. 현재
science namelist는 `rmerge=4`, `vrel_merge=.true.`, `drag=.true.`이다.

이 구현은 다음을 계산하지 않는다.

- hard binary 형성
- stellar loss-cone evolution
- sub-pc FDM wave drag
- gravitational radiation
- 수치 병합 뒤의 physical delay time

현재 로그도 보존되는 pair별 사건 원장 대신 `Found N groups from M` 정도만 출력하므로 정확한
후처리를 위해 별도 merger-event output이 필요하다.

### 수치적 병합 반경

코드에서

\[
dx_{\min}=\frac{L_{\rm code}\,2^{-\ell_{\max}}}{a},\qquad
{\tt scale_l}=\frac{aL_{\rm box}}{h}
\]

이므로 physical merge radius에서 `a`가 상쇄된다.

\[
r_{\rm num}=r_{\rm merge}\frac{L_{\rm box}/h}{2^{\ell_{\max}}}.
\]

`h=0.6766`, `rmerge=4`를 적용하면 다음과 같다.

| run family | box | levelmax | one sink dx | numerical merge radius |
|---|---:|---:|---:|---:|
| matched-v4 | 1.2 Mpc/h | 12 | 433 pc | 1.732 kpc |
| clean 64 Mpc | 64 Mpc/h | 14 | 5.773 kpc | 23.093 kpc |
| clean 128 Mpc | 128 Mpc/h | 15 | 5.773 kpc | 23.093 kpc |

따라서 문서와 논문에서 `sink merger`, `binary formation`, `physical coalescence`를 서로 바꾸어
쓰면 안 된다.

### 2026-08-09 현재 matched-v4 sink 상태

2026-08-09 07:49 KST에 latest complete output을 확인했다.

| arm | output | a | sink count | sink mass |
|---|---:|---:|---:|---:|
| wave | 38 | 0.39814 | 0 | - |
| hybrid | 26 | 0.40329 | 1 | 약 `1.5589e4 Msun` |
| fluid | 26 | 0.40429 | 1 | 약 `1.8384e4 Msun` |

현재 binary 또는 실제 sink-merger 사건은 없다. 기존 완료 64/128 Mpc fluid의 보존 output에서도
두 sink가 합쳐진 사건을 재구성할 자료는 확인되지 않았다. 따라서 첫 toy model은 synthetic 및
문헌 benchmark로 검증해야 한다.

---

## 4. 문헌에서 얻는 물리와 제한

### Koo et al. 2024

- `M_sol = 1e9 Msun`
- 각 SMBH 질량 `0.6, 1.0, 1.5e8 Msun`
- 평균 초기 separation 약 `0.8 pc`
- wave dynamical friction과 gravitational cooling으로 빠른 초기 decay를 관측
- calibration `alpha_DF ~= 0.424`
- `M_BH=1e8 Msun` 예에서 초기 decay scale 약 `0.47--0.62 Myr`
- **0.01 pc까지 full decay simulation은 수행하지 않았음**

논문: <https://arxiv.org/abs/2311.03412>

### Boey et al. 2025

- AxioNyx로 SMBH binary와 soliton을 더 길고 높은 해상도로 계산
- SMBH 질량 `2e7--1e8 Msun`, 대표 soliton `1e9 Msun`
- 접근하는 SMBH가 soliton을 pinch하여 중심밀도가 초기값의 약 5배까지 상승할 수 있음
- fiducial model에서 `1 pc -> 0.076 pc`를 약 `5.6 Myr`로 추정
- `2e7 Msun` BH의 `1 pc -> 0.023 pc` 예는 약 `477 Myr`
- final-parsec 완화 가능성은 주로 `m_FDM >~ 1e-21 eV`에서 제시
- empirical fit은 `D=A(1+Bt)^(-C)` 형태

대표 fit table:

| each-BH / soliton mass ratio | A [pc] | B [Myr^-1] | C |
|---:|---:|---:|---:|
| 2% | 2.94 | 3.12 | 0.662 |
| 5% | 2.78 | 7.44 | 0.777 |
| 10% | 2.72 | 22.7 | 0.733 |

논문: <https://arxiv.org/abs/2504.16348>

### 우리 계산에 대한 직접 경고

우리 science run은 `m_FDM=1e-22 eV`, seed mass `1e4 Msun`이다. 문헌 calibration은 주로
`m_FDM~1e-21 eV`, `M_BH~1e7--1e8 Msun`에 집중한다. 그러므로 문헌의 Myr-scale decay를 우리
simulation에 그대로 대입하면 안 된다.

고정 soliton mass에서는 core radius가 대략 `m_FDM^-2`, 중심밀도가 `m_FDM^6`으로 변하고 drag
coefficient에도 별도 `m_FDM^2` 의존성이 있어 민감도가 매우 크다. `m=1e-22 eV` 계산이
Hubble time 안에 0.01 pc에 도달하지 못하는 결과도 정상 출력으로 인정해야 한다.

Lancaster et al.은 다음 비선형성 지표가 1에 가까우면 단순 analytic drag가 신뢰되기 어렵다고
경고한다.

\[
\eta_{\rm nl}=\left(\frac{M}{10^9M_\odot}\right)
\left(\frac{m}{10^{-22}{\rm eV}}\right)
\left(\frac{100\,{\rm km\,s^{-1}}}{v}\right).
\]

논문: <https://arxiv.org/abs/1909.06381>

원형궤도 linear-response 계산에서는 순간 torque가 drag가 아니라 thrust로 바뀌거나 inspiral이
stall할 가능성도 있다. 첫 toy model은 orbit-averaged monotonic drag model임을 명시하고 이
한계를 validity flag에 넣는다.

참조: <https://arxiv.org/abs/2207.13740>

---

## 5. V0 toy model의 상태와 입력

### 필수 입력

- `M1`, `M2`: 각 SMBH mass
- `r1`, `r2`: soliton 중심 기준 3D 위치
- `v1`, `v2`: local FDM bulk velocity 기준 3D 속도
- `m_fdm`: FDM particle mass
- `M_soliton`: soliton 질량과 그 정의
- soliton profile 중 하나:
  - `rho0 + rc`
  - tabulated `r, rho, M_enclosed`
  - `M_soliton + rc`와 명시적 profile normalization
- `alpha_df`: effective cutoff calibration
- `D_stop=0.01 pc`

### 권장 선택 입력

- initial eccentricity 또는 orbital phase
- soliton pinching factor
- local FDM bulk velocity
- 시작 시각/redshift
- 최대 적분시간: 기본 Hubble time 또는 사용자가 지정한 cosmic-time budget

### 독립 입력으로 받지 말아야 할 값

`total energy`는 질량, 위치, 속도와 potential이 주어지면 결정된다. 입력 파일에 energy가 함께
들어오면 초기조건 consistency check에만 사용한다. 서로 불일치하면 실행을 중단한다.

---

## 6. V0 물리식

### 좌표

\[
\mathbf D=\mathbf r_1-\mathbf r_2,\qquad
D=|\mathbf D|,\qquad
\mathbf v_{\rm rel}=\mathbf v_1-\mathbf v_2.
\]

각 SMBH의 FDM 상대속도는 binary COM 속도가 아니라 local FDM bulk flow를 뺀

\[
\mathbf w_i=\mathbf v_i-\mathbf u_{\rm FDM}(\mathbf r_i)
\]

를 사용한다. V0 static soliton에서는 `u_FDM=0`이다.

### Soliton density와 potential

기본 profile은

\[
\rho(r)=\rho_0\left[1+0.091(r/r_c)^2\right]^{-8}
\]

로 시작한다. `M_soliton` 정의가 논문마다 다르므로 coefficient를 하드코딩하지 않는다. config에
`mass_definition = total_profile | within_rc | tabulated`를 의무화하고 numerical quadrature로
normalization과 `M_enclosed(r)`를 만든다.

Soliton acceleration은

\[
\mathbf a_{\rm sol}(\mathbf r)
=-\frac{G M_{\rm enc}(r)}{r^3}\mathbf r
\]

이다. SMBH가 soliton profile을 변화시키는 pinching은 V0의 주 물리식에 섞지 않는다.

### FDM drag

각 SMBH에 다음 orbit-averaged force를 적용한다.

\[
\mathbf F_{{\rm DF},i}
=-4\pi G^2M_i^2\rho(r_i)C(q_i)
\frac{\mathbf w_i}{|\mathbf w_i|^3}.
\]

\[
q_i=\frac{m_{\rm FDM}|\mathbf w_i|r_{{\rm eff},i}}{\hbar},
\qquad
r_{\rm eff}=\alpha_{\rm DF}\min(D/2,r_c).
\]

점질량, uniform-background wave response는

\[
C(q)=\operatorname{Cin}(2q)+\frac{\sin(2q)}{2q}-1,
\qquad
\operatorname{Cin}(x)=\int_0^x\frac{1-\cos t}{t}\,dt
\]

로 두고, `q << 1`에서는 cancellation을 피하기 위해

\[
C(q)=q^2/3+O(q^4)
\]

series를 사용한다. `q`, `M_enc/M_BH`, `eta_nl`을 매 timestep 기록한다.

### 운동방정식

\[
\ddot{\mathbf r}_1=
-GM_2\frac{\mathbf r_1-\mathbf r_2}{D^3}
+\mathbf a_{\rm sol}(\mathbf r_1)
+\frac{\mathbf F_{{\rm DF},1}}{M_1},
\]

\[
\ddot{\mathbf r}_2=
-GM_1\frac{\mathbf r_2-\mathbf r_1}{D^3}
+\mathbf a_{\rm sol}(\mathbf r_2)
+\frac{\mathbf F_{{\rm DF},2}}{M_2}.
\]

처음에는 static spherical soliton에서 두 SMBH의 3D 궤도를 직접 적분한다. 이후 필요하면
orbit-averaged `dE/dt`, `dL/dt` 모드를 추가한다.

### 에너지 원장

\[
E_{\rm mech}=
\sum_i\frac12M_iv_i^2
-\frac{GM_1M_2}{D}
+\sum_iM_i\Phi_{\rm sol}(r_i).
\]

FDM으로 전달된 누적 에너지는

\[
E_{\rm FDM}(t)
=-\int_0^t\sum_i\mathbf F_{{\rm DF},i}\cdot\mathbf w_i\,dt.
\]

static-background V0에서는

\[
E_{\rm budget}=E_{\rm mech}+E_{\rm FDM}
\]

의 drift를 수치 오차로 사용한다. `rho`를 separation에 따라 임의로 변화시키는 pinching model은
time-dependent background work를 추가하므로 V0 energy-conservation test와 분리한다.

### 종료 조건

- 성공: inward crossing에서 `D <= 0.01 pc`
- 실패/검열 자료: `t >= t_max`인데 도달하지 못함
- fatal: NaN, unbound escape, negative density/mass, energy residual 초과
- validity warning: analytic DF 적용범위 위반

단순 pericentre passage를 coalescence로 오인하지 않도록 inward crossing과 bound-orbit 조건을
함께 검사한다.

---

## 7. V0 출력 형식

### Summary JSON

```json
{
  "model_version": "0.1.0",
  "input_hash": "...",
  "status": "reached_0p01pc | timeout | unbound | invalid",
  "t_fdm_myr": 0.0,
  "D_initial_pc": 1.0,
  "D_final_pc": 0.01,
  "E_initial": 0.0,
  "E_final": 0.0,
  "E_to_fdm": 0.0,
  "max_energy_budget_relerr": 0.0,
  "max_q": 0.0,
  "max_eta_nl": 0.0,
  "validity_flags": []
}
```

### Time-series columns

```text
t, D, r1[3], r2[3], v1[3], v2[3],
E_kin, E_bh_bh, E_soliton, E_mech, E_to_fdm, E_budget,
L[3], eccentricity_osculating, rho1, rho2, q1, q2, eta_nl
```

output에는 항상 input config, source commit, dependency versions, solver tolerance와 단위계를 함께
저장한다.

---

## 8. 두 개의 보조 모델

### Koo analytic cross-check

\[
D(t)=D_0\left[1+\frac52Q_0D_0^{5/2}t\right]^{-2/5}.
\]

이 식은 small-separation, near-circular cross-check로만 사용한다. main integrator가 아니다.

### Boey empirical cross-check

\[
D(t)=A(1+Bt)^{-C}.
\]

두 separation 사이의 시간은

\[
\Delta t=\frac{1}{B}
\left[
\left(\frac{A}{D_f}\right)^{1/C}
-\left(\frac{A}{D_i}\right)^{1/C}
\right].
\]

문헌과 동일한 parameter에서만 benchmark로 사용한다. `m_FDM`, `M_BH`, `M_soliton`이 calibration
domain 밖이면 empirical 결과에 `EXTRAPOLATED` flag를 붙인다.

---

## 9. 검증 기준

### 필수 regression

1. `drag=0`, static soliton에서 bound orbit의 energy와 angular momentum 수렴
2. timestep/tolerance를 절반으로 줄였을 때 `t_FDM` 수렴
3. static drag에서 `E_mech + E_FDM` 상대 drift 기준 통과
4. small-`q` series와 full `C(q)`가 overlap 영역에서 일치
5. equal-mass circular orbit에서 COM drift가 수치 오차 수준

### 문헌 benchmark

1. Koo의 `M_BH=1e8 Msun`, `M_sol=1e9 Msun`, `D~0.8 pc` 초기 decay scale을 재현
2. Boey fiducial의 `1 pc -> 0.076 pc ~ 5.6 Myr`을 허용범위 안에서 재현
3. Boey의 BH/soliton mass ratio 2%, 5%, 10% 순서에 따른 decay-rate 증가를 재현
4. `m_FDM`을 키울 때 decay가 급격히 빨라지는 정성적 trend 재현

문헌 simulation의 full density history를 갖고 있지 않으므로 정확한 일치보다 공개된 fit을 같은
입력으로 재생하는 test와 ODE의 trend test를 구분한다.

### 제안 허용오차

- no-drag energy drift: `<1e-8`
- drag energy-budget drift: `<1e-6`
- tolerance-halving `t_FDM` 변화: `<1%`
- 공개 empirical fit 재생: machine precision
- 문헌 수치 benchmark: 첫 목표 `20%` 이내, profile 정의를 맞춘 뒤 `10%` 이내

---

## 10. 첫 parameter scan

### Calibration set

- `m_FDM = 1e-21 eV`
- `M_soliton = 1e9 Msun`
- equal BH masses `2e7, 5e7, 1e8 Msun`
- circular orbit, `D0=1 pc`
- `alpha_DF = 0.3027, 0.341, 0.424`
- `pinch_factor = 1, 5`는 별도 bracket

### 우리 과학 set

- `m_FDM = 1e-22 eV`
- 실제 lagRamses에서 예상되는 `M1`, `M2` 범위
- `M_soliton = 1e7--2e9 Msun`
- mass ratio `q_BH = 0.1, 0.3, 1`
- eccentricity `e = 0, 0.3, 0.7`
- `D0=1 pc`, `Dstop=0.01 pc`
- timeout을 Hubble time과 사건별 남은 cosmic time 두 가지로 평가

seed mass `1e4 Msun`까지 scan할 수는 있으나 문헌 calibration 밖이라는 점을 명확히 표시한다.
작은 BH에서 timeout은 실패가 아니라 물리적 결과다.

---

## 11. 권장 software 구조

사용자가 새 디렉터리를 만든 뒤 다음 구조를 권장한다.

```text
FDM_SMBH_DELAY/
├── README.md
├── pyproject.toml
├── CITATION.cff
├── src/fdm_smbh_delay/
│   ├── __init__.py
│   ├── constants.py
│   ├── units.py
│   ├── soliton.py
│   ├── wave_drag.py
│   ├── orbit.py
│   ├── empirical.py
│   ├── validity.py
│   └── io.py
├── configs/
│   ├── koo2024.yaml
│   ├── boey2025_fiducial.yaml
│   └── lagramses_m22_example.yaml
├── scripts/
│   ├── run_case.py
│   ├── run_grid.py
│   └── reproduce_literature.py
├── tests/
│   ├── test_units.py
│   ├── test_no_drag_orbit.py
│   ├── test_energy_budget.py
│   ├── test_drag_coefficient.py
│   ├── test_koo2024.py
│   └── test_boey2025.py
├── docs/
│   ├── equations.md
│   ├── validity.md
│   └── lagramses_interface.md
└── results/
```

Jupyter notebook를 원장으로 쓰지 않는다. 논문 figure notebook가 필요하더라도 모든 핵심 계산은
import 가능한 module과 CLI로 재현되게 한다.

### 단위 정책

- 외부 interface: `Msun`, `pc`, `Myr`, `km/s`, `eV`
- config parse 단계에서 unit-aware validation
- integrator 내부는 하나의 고정 단위계만 사용
- `G`, `hbar`, `c`, eV-to-mass 변환을 한 파일에 고정
- unitless float를 public API에 조용히 허용하지 않음

---

## 12. 예시 config

```yaml
model:
  name: wave_df_3d
  alpha_df: 0.341
  drag_coefficient: hui_full
  fdm_bulk_velocity: [0.0, 0.0, 0.0]
  pinch_factor: 1.0

binary:
  M1: "5.0e7 Msun"
  M2: "5.0e7 Msun"
  separation: "1.0 pc"
  eccentricity: 0.0
  orbit: circular

fdm:
  particle_mass: "1.0e-21 eV"
  soliton_mass: "1.0e9 Msun"
  mass_definition: total_profile
  core_radius: "2.0 pc"
  profile: schive_fit

integration:
  stop_separation: "0.01 pc"
  max_time: "14 Gyr"
  rtol: 1.0e-9
  atol: 1.0e-12
```

`core_radius`와 density normalization은 문헌 case별 정의를 확인해 넣어야 한다. 위 숫자는 schema
예시이며 검증된 fiducial 값이 아니다.

---

## 13. 향후 lagRamses merger-event interface

### 원칙

lagRamses는 수치 병합 직전 상태를 기록하고, 새 프로젝트는 이를 읽어 physical delay를 계산한다.
첫 단계에서는 delay를 simulation dynamics에 feedback하지 않는다.

### 필수 event fields

```text
provenance:
  code_commit, binary_sha256, job_id, run_id, step, aexp, cosmic_time

binary:
  group_id, member_count, sink_id_1, sink_id_2
  M_dyn_1, M_dyn_2, M_bh_1, M_bh_2
  position_1[3], position_2[3]
  velocity_1[3], velocity_2[3]
  spin_1[3], spin_2[3]
  separation_physical, relative_velocity
  E_kin_com, E_grav_pair, angular_momentum

environment:
  rho_gas, sound_speed, gas_bulk_velocity
  rho_star, stellar_velocity_dispersion
  rho_fdm, fdm_bulk_velocity
  halo_id, halo_mass
  soliton_mass, core_radius, central_density, fit_quality
```

### 구현 위치

`merge_sink`가 FOF compaction으로 member 정보를 지우기 직전 event를 기록한다. 세 개 이상의 sink가
한 group으로 묶이면 임의의 pair sequence로 해석하지 말고 `MULTIPLE` flag와 전체 member list를
저장한다.

### 현재 출력의 한계

현재 snapshot sink table은 살아남은 sink만 기록하고 output 간격도 병합 순간보다 길 수 있다.
따라서 sink ID가 사라진 snapshot만으로 exact separation, velocity, local density를 복원하는 방식은
최종 분석에 충분하지 않다.

---

## 14. 구현 순서

### P0: 독립 project bootstrap

1. repository와 package skeleton 생성
2. 단위계, config schema, provenance output 구현
3. soliton profile과 enclosed-mass numerical integration 구현
4. full/small-q drag coefficient 구현

### P1: V0 integrator

1. no-drag two-body+static-soliton orbit
2. wave drag와 energy ledger 결합
3. `1 pc -> 0.01 pc` event detection
4. timeout/unbound/invalid 상태 구현

### P2: 문헌 재현

1. Koo analytic curve
2. Boey empirical curve와 table
3. ODE 결과와 두 curve 비교
4. 공개 calibration domain metadata 고정

### P3: 우리 `m22=1` scan

1. BH mass, mass ratio, soliton mass, eccentricity scan
2. `alpha_DF` 및 pinching uncertainty bracket
3. Hubble-time censoring
4. 결과 surrogate/table 구축

### P4: lagRamses 연결

1. event schema 확정
2. 별도 lagRamses branch에서 logger 구현
3. grammar-debug synthetic two-sink test
4. 미래 science run부터 event catalogue 생성

### P5: 나머지 두 구간

1. `r_num -> 1 pc` gas/star/FDM pairing model
2. `<0.01 pc` Peters/PN GW module
3. 세 구간의 uncertainty propagation
4. merger-rate 및 redshift distribution 계산

---

## 15. 하지 말아야 할 것

- 현재 실행 중인 matched-v4 binary를 이 프로젝트 때문에 교체하지 않는다.
- `Found groups` 시각을 physical SMBH coalescence로 부르지 않는다.
- Boey의 `m~1e-21 eV` Myr-scale 결과를 우리 `m=1e-22 eV`에 그대로 대입하지 않는다.
- `1 pc -> 0.01 pc` 모델로 1.7--23 kpc 구간까지 외삽하지 않는다.
- soliton mass 정의와 core-radius 정의를 생략하지 않는다.
- static analytic drag가 nonlinear wave response를 정확히 나타낸다고 주장하지 않는다.
- timeout case를 계산 실패로 버리지 않는다.
- pinching factor를 조절해 원하는 merger time에 맞추지 않는다.
- 검증 전 toy model을 lagRamses accretion, feedback, spin 또는 recoil에 coupling하지 않는다.

---

## 16. Definition of Done: 첫 release

다음을 모두 만족하면 `v0.1`이 완료된 것으로 본다.

- clean environment에서 package 설치 및 test 실행 가능
- arbitrary equal/unequal-mass 3D initial state 입력 가능
- static soliton과 full/small-q FDM drag 지원
- `1 pc -> 0.01 pc` 도달시간 또는 timeout 반환
- energy transfer ledger와 validity flags 출력
- no-drag 및 timestep convergence test 통과
- Koo/Boey 공개 fit 재생 test 통과
- Boey fiducial benchmark 비교 문서화
- `m_FDM=1e-22 eV` 최소 parameter grid 완료
- 모든 결과가 config hash와 code commit을 보존
- README에 “physical prediction이 아니라 calibrated toy/subgrid model”이라고 명시

---

## 17. 열린 과학적 결정

1. `M_soliton`을 total profile mass, core mass, 또는 fitted inner mass 중 무엇으로 표준화할 것인가?
2. current FDM halo에서 `rho0`, `rc`를 어느 반경 범위로 fit할 것인가?
3. unresolved SMBH가 soliton을 pinch하는 효과를 density multiplier로 둘지 별도 response model로
   만들 것인가?
4. eccentric orbit에서 instantaneous wave force와 orbit-averaged force 중 어느 것을 기준으로 할 것인가?
5. FDM bulk velocity/phase gradient를 future lagRamses output에서 어떻게 정의할 것인가?
6. gas/star drag와 FDM drag 사이의 transition radius를 고정할지 사건별로 정할 것인가?
7. simulation에서는 sink를 즉시 합친 채 delay를 catalogue에만 기록할지, 장기적으로 unresolved
   binary object를 유지할 것인가?

첫 release에서는 1, 3, 4를 config option으로 노출하고, 하나의 값을 정답처럼 고정하지 않는다.

---

## 18. 핵심 참고문헌

1. Koo, Bak, Park, Hong & Lee, *Final parsec problem of black hole mergers and ultralight dark matter*,
   Physics Letters B 856, 138908 (2024): <https://arxiv.org/abs/2311.03412>
2. Boey, Kendall, Wang & Easther, *Supermassive Binaries in Ultralight Dark Matter Solitons*,
   Physical Review D 112, 023510 (2025): <https://arxiv.org/abs/2504.16348>
3. Hui, Ostriker, Tremaine & Witten, *Ultralight scalars as cosmological dark matter*,
   Physical Review D 95, 043541 (2017): <https://arxiv.org/abs/1610.08297>
4. Lancaster et al., *Dynamical Friction in a Fuzzy Dark Matter Universe*,
   JCAP 01, 001 (2020): <https://arxiv.org/abs/1909.06381>
5. Buehler & Desjacques, *Dynamical friction in fuzzy dark matter: circular orbits*,
   Physical Review D 107, 023516 (2023): <https://arxiv.org/abs/2207.13740>
6. Wang & Easther, *Dynamical Friction From Ultralight Dark Matter*,
   Physical Review D 105, 063523 (2022): <https://arxiv.org/abs/2110.03428>
7. Davies & Mocz, *Fuzzy Dark Matter Soliton Cores around Supermassive Black Holes*,
   Monthly Notices of the Royal Astronomical Society 492, 5721 (2020):
   <https://arxiv.org/abs/1908.04790>

---

## 19. 새 디렉터리에서 가장 먼저 할 일

```text
[ ] 이 인수인계서를 새 project의 docs/HANDOFF.md로 복사
[ ] Git repository 초기화
[ ] README에 구간 2만 v0.1 scope로 고정
[ ] pyproject와 unit policy 작성
[ ] Koo/Boey benchmark config부터 작성
[ ] drag coefficient unit test 작성
[ ] no-drag orbit와 energy test 작성
[ ] 그 뒤에만 dissipative ODE를 구현
```

첫 과학 결과는 우리 simulation에 서둘러 적용한 merger time이 아니라, 문헌 benchmark를 재현한
뒤 `m22=1`에서 어떤 SMBH/soliton 조합이 Hubble time 안에 0.01 pc에 도달하는지를 보여주는
parameter map이어야 한다.
