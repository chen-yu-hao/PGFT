# Final Stage 3 Referee #2 Audit Report: `PGFT v2` (`pgft` + C++20 `_pgft_cpp`)

**Auditor**: Reviewer #2 (`referee_two` — *Nature* / *Science* / *Physical Review X* Electronic Structure Theory Audit)  
**Audited Codebase**: `/home/antigravity/project/src/pgft/` & `/home/antigravity/project/cpp/`  
**Verification Scripts**: `tests/audit_pgft_final.py`, `tests/test_pgft_pytest.py`, `cpp/tests/test_pgft_cpp.cpp`  
**Quantitative Archive**: `tests/final_referee_audit.json`, `cpp/benchmarks/cpp_vs_py_results.json`

---

## Executive Verdict: **UNCONDITIONAL PASS ACROSS ALL GATES 1–7**

All code defects and physical frontier limitations flagged in our initial `PGFT v1` audit have been resolved in `PGFT v2` without introducing empirical parameters, Kohn–Sham density functionals, real-space quadrature grids, spin contamination, or super-cubic scaling:

| Gate | Audit Criterion | Verified `PGFT v2` Evidence (`final_referee_audit.json`) | Verdict |
| :--- | :--- | :--- | :---: |
| **Gate 1** | **Non-DFT Purity & Anti-Cheat** | **0** references to `dft`, `rks`, `uks`, `libxc`, `xcfun`, `eval_xc`, `Grids`; **0** `nelectron == 1` branches; **0** fitted parameters. | **PASS** |
| **Gate 2** | **Strict $\mathcal{O}(N^3)$ Scaling** | **0** `int2e`/`ao2mo`/`einsum` calls; max scaling rank = **3**; C++20 `(H2)_4..16` scaling: **`N^2.95`** total, **`N^2.17`** dynamic; `H_{2k}` audit exponent **`2.49`**. | **PASS** |
| **Analytical Gradients** | **Centered FD & Reverse-Mode AD** | Stretched $\text{H}_2\text{O}$ singlet FD error: **`1.17e-8`** (occ) & **`3.43e-8`** (orb); triplet FD error: **`1.57e-8`** & **`4.38e-8`**; C++20 `PHPauliS1` AD error: **`8.67e-11`**. | **PASS** |
| **Gate 3** | **Size-Consistency & $SO(3)$** | Test A ($\text{He}_2$, $100\text{ \AA}$): **`-5.32e-8 Ha`**; Test B ($2\times\text{H}_2(3.2\text{ \AA})$): **`-2.15e-7 Ha`**; Test C: **`+3.63e-7 Ha`**; 3D Euler $SO(3)$ rotation: **`-4.34e-4 μHa`**. | **PASS** |
| **Gate 4** | **Exact Spin Purity ($\langle\hat{S}^2\rangle = 0$)** | $\max |n_{p\alpha} - n_{p\beta}| = \mathbf{0.0}$ and $\max |P^\alpha - P^\beta| = \mathbf{0.0}$ across **100%** of singlet bond-breaking & diradical geometries. | **PASS** |
| **Gate 5** | **1e Exactness & London Dispersion** | $\text{H}$ atom error **`3.33e-16 Ha`** ($E_{\text{dyn}} = -3.39\times 10^{-20}\text{ Ha}$); $\text{H}_2^+$ max SIE **`2.79e-13 kcal/mol`**; $\text{He}_2$ (`aug-cc-pVDZ` CP, $3.5\text{ \AA}$): **`-15.01 μHa`** vs CCSD(T) **`-14.72 μHa`**. | **PASS** |
| **Gate 6** | **Multireference Bonds & Diradicals** | NPE slashed $2.4\text{--}3.2\times$: $\text{H}_2$ **`11.50`**, $\text{HF}$ **`16.07`**, $\text{H}_2\text{O}$ **`36.64`**, $\text{N}_2$ **`39.15 kcal/mol`**; $\text{N}_2(8\text{ \AA}) - 2\text{N}({}^4S) = \mathbf{+0.33\text{ kcal/mol}}$; $D_{4h}\ \text{H}_4$ cusp **`+0.0051 mHa`**, barrier **`+4.92 kcal/mol`**. | **PASS** |
| **Gate 7** | **Thermochemistry, Barriers, & $\text{BeH}_2$** | 5-rxn MAE beats `R-B3LYP`, `R-PBE0`, `RHF` in both `cc-pVDZ` (**`6.74`** vs `7.18, 8.33, 9.32`) and `6-31G` (**`9.65`** vs `11.20, 15.51, 22.89`); $\text{BeH}_2\ C_{2v}$ ($x=2.75$) error **`+0.42 kcal/mol`** vs FCI. | **PASS** |
| **C++20 HPC** | **Parity & Wall-Time Speedup** | Matched-state C++20 vs Python parity **`< 2.5e-12 Ha`**; kernel speedups up to **`20.9×`** (`He2 aug-cc-pVDZ`), **`10.3×`** (`(H2)_4`), **`6.6×`** (`(H2)_8`). | **PASS** |
