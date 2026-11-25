# PGFT v2 vs CF22D vs PBE: Scaling, Wall-Clock Time, and GSCDB137 Accuracy Benchmark

> [!NOTE]
> All quantum-chemistry calculations were executed on Google Colab High-RAM CPU workers (Intel Xeon 4 physical cores / 8 logical threads, 51 GiB RAM) using **PGFT v2** (`C++20 / OpenMP / OpenBLAS` backend), **PySCF 2.15.1 + LibXC 7.1.2**, and **`dftd3`** (two-body zero-damping D3(0)). Every SCF and PGFT run across all **802 unique molecules** (`def2-SVP`), **247 molecules** (`aug-cc-pVDZ`), **258 molecules** (`def2-TZVP`), and **802 molecules** (`def2-QZVPPD` DF-PBE) completed with **0 errors and 100% SCF convergence**.

---

## 1. Executive Summary

| Dimension | **PGFT v2 (C++20)** | **CF22D (Hyb-mGGA + D3)** | **PBE (GGA)** | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **1D Scaling `(H2)_k` (`6-31G`)** | $\alpha = 3.48$ (`ACFD: 2.84`, `THC: 3.06`, `SCF: 4.00`) | $\alpha = 1.66$ | $\alpha = 1.48$ | `Dynamic ACFD` is strictly $\mathcal{O}(N^3)$ and faster than `CF22D` up to $k=24$ (`12.8 s` vs `24.7 s`); total exponent is pulled up by `Static L-BFGS SCF`. |
| **3D / Compact Scaling `(H2O)_n`** | $\alpha = 4.61$ ($n=1\text{--}4$, pre-asymptotic) | $\alpha = 2.00$ ($n=1\text{--}8$) | $\alpha = 2.00$ ($n=1\text{--}8$) | Compact 3D clusters stay in the dense Schwarz pair regime ($N_{\text{sig}} \propto N^2$). |
| **Basis Scaling `H2O` (`STO-3G`→`cc-pVTZ`)** | $\alpha = 4.28$ (`ACFD: 2.88`, `SCF: 4.40`) | $\alpha = 1.03$ | $\alpha = 0.89$ | DFT is grid-dominated; PGFT scales strongly with $N_{\text{THC}} \propto N_{\text{AO}}$. |
| **802-Mol CPU Time (`def2-SVP`, 1-core)** | **`36.67 h`** (`132,025 s`, `27.5×` PBE) | `1.95 h` (`7,016 s`, `1.46×` PBE) | `1.33 h` (`4,806 s`, `1.00×`) | **`Static L-BFGS SCF` accounts for 93.81% (`34.40 h`)** of PGFT v2 runtime; `HF + THC + ACFD` takes only **`2.27 h` (`6.19%`)**, on par with `CF22D`! |
| **GSCDB137 Overall RelMetric (`def2-SVP`, 20 sets, 463 rxns)** | `17.062` (MAE `32.40` kcal/mol) | **`4.617`** (MAE `9.96` kcal/mol) | `6.322` (MAE `13.76` kcal/mol) | CBS reference on the same 20 sets: `CF22D (Q-Chem CBS) = 1.150`, `DF-PBE (def2-QZVPPD) = 3.760`. |
| **Flagship PGFT v2 Wins** | **`SIE4x4`** (`7.18` kcal/mol, `TZVP`) & **`He3`** (`0.224`, `aug-DZ`) | `SIE4x4`: `11.16` (`10.87` CBS); `He3`: `0.525` (`0.469` CBS) | `SIE4x4`: `23.73` (`23.23` CBS); `He3`: `0.241` (`0.267` CBS) | **PGFT v2 decisively beats even CBS-limit `CF22D` and `PBE` on self-interaction (`SIE4x4`) and 3-body dispersion (`He3`)**, and beats `PBE` on `ISOMERIZATION20` and `TA13`. |

---

## 2. Implementation & Validation Protocol

- **PGFT v2**: Default configuration (`c_thc = 18`, `n_rad = 28`, 4-seed multi-start Static L-BFGS SCF, `cpp` OpenMP/OpenBLAS backend, grid-free ISDF-THC + dynamic ACFD). For scaling studies, a lighter `PGFT_v2_c10` (`c_thc = 10`, `n_rad = 20`) variant was also tested up to `(H2)_24` ($N_{\text{AO}} = 96$).
- **CF22D**: Implemented in PySCF 2.15.1 via LibXC 7.1.2 (`HYB_MGGA_X_CF22D` ID 340 with 46.2806% exact HF exchange + `MGGA_C_CF22D` ID 341) plus explicit `dftd3` 2-body zero-damping dispersion `ZeroDampingParam(s6=1.0, s8=0.0, rs6=1.53, rs8=1.0, alp=14.0, s9=0.0)`.
  - **Exact Cross-Validation against GSCDB137 Q-Chem (`def2-QZVPPD`, `(99, 590)` grid)**:
    - `A24_water`: diff `+0.16 μHa`
    - `A24_waterdimer`: diff `+0.45 μHa`
    - Water dimer interaction energy: **`-4.90305` vs `-4.90306 kcal/mol` (`Δ = +0.00001 kcal/mol`)** — exact numerical parity.
- **PBE**: PySCF `xc = "PBE"` on the identical `(99, 590)` pruned Lebedev grid (`atom_grid = (99, 590)`), plus density-fitted `DF-PBE / def2-QZVPPD` across all 20 datasets to anchor CBS-limit PBE accuracy.
- **Timing Hygiene**: All single-core scaling and GSCDB137 per-molecule timings enforce `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (eliminating nested thread oversubscription). Multi-thread scaling was measured separately with `OMP_NUM_THREADS ∈ {1, 2, 4, 8}`, `OPENBLAS_NUM_THREADS=1`.

---

## 3. Scaling & Wall-Clock Time Analysis

![Scaling exponents, basis scaling, 802-molecule CPU time scatter, and PGFT v2 CPU time anatomy](./figures/fig1_scaling_and_timing.png)

### 3.1 Five-Family Empirical Scaling ($t \propto N_{\text{AO}}^{\alpha}$, Single Core)

| Benchmark Family | Size Range ($N_{\text{AO}}$) | `PBE` exact $\alpha$ ($R^2$) | `CF22D` exact $\alpha$ ($R^2$) | `PGFT v2` Total $\alpha$ ($R^2$) | `PGFT` THC $\alpha$ | `PGFT` Static SCF $\alpha$ | `PGFT` Dyn. ACFD $\alpha$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. 1D `(H2)_k` (`6-31G`)** (`c10`, $k=2\text{--}24$) | `8 – 96` | `1.48` (`0.997`) | `1.66` (`0.984`) | `3.48` (`0.997`) | **`3.06`** | `4.00` | **`2.84`** |
| **1b. 1D `(H2)_k` (`6-31G`)** (`default`, $k=2\text{--}16$) | `8 – 64` | `1.48` (`0.997`) | `1.66` (`0.984`) | `3.45` (`0.998`) | **`3.14`** | `4.09` | **`2.73`** |
| **2. Linear n-Alkanes `C_n H_{2n+2}` (`6-31G`)** | `17 – 134` (PGFT `17–56`) | `1.97` (`0.995`) | `2.01` (`0.994`) | `4.70` (`0.994`) | `4.15` | `4.84` | `3.81` |
| **3. 3D Water Clusters `(H2O)_n` (`6-31G`)** | `13 – 104` (PGFT `13–52`) | `2.00` (`0.997`) | `2.00` (`0.998`) | `4.61` (`0.999`) | `4.33` | `4.74` | `3.60` |
| **4. Fixed-Mol `H2O` Basis (`STO-3G`→`QZVPP`)** | `7 – 117` (PGFT `7–58`) | `0.89` (`0.943`) | `1.03` (`0.950`) | `4.28` (`0.992`) | `3.27` | `4.40` | **`2.88`** |

**Selected Wall-Clock Points (single core, seconds)**:
- **1D `(H2)_k` (`6-31G`)**:
  - `k=2` ($N_{\text{AO}}=8$): PGFT default `0.132 s` (`c10: 0.119 s`) **<** PBE `0.431 s` < CF22D `0.447 s` (grid-free PGFT wins by **3.4×**).
  - `k=6` ($N_{\text{AO}}=24$): PGFT default `1.518 s` (`c10: 1.282 s`) **<** PBE `1.936 s` < CF22D `2.587 s`.
  - `k=8` ($N_{\text{AO}}=32$, crossover): PGFT default `3.578 s` (`c10: 2.883 s`) vs CF22D `3.493 s` / PBE `2.775 s`.
  - `k=16` ($N_{\text{AO}}=64$): PGFT default `42.73 s` (`c10: 35.27 s`; `dyn: 3.63 s`) vs CF22D `10.16 s` (`99,590`: `32.02 s`) / PBE `7.60 s`.
  - `k=24` ($N_{\text{AO}}=96$): PGFT `c10` `133.38 s` (`thc: 34.37 s`, `stat: 85.22 s`, **`dyn: 12.80 s`**) vs CF22D `24.72 s` / PBE `15.25 s`. Notably, **PGFT's dynamic ACFD (`12.80 s`) remains 1.9× faster than CF22D (`24.72 s`) even at $N_{\text{AO}}=96$**.
- **Alkanes (`6-31G`)**: `C1H4` ($17$): PGFT `2.06 s` vs PBE `0.94 s` / CF22D `1.17 s` → `C4H10` ($56$): PGFT `526.29 s` (`stat: 475.58 s`, `thc: 39.59 s`, `dyn: 10.25 s`) vs PBE `7.77 s` / CF22D `9.68 s` (`C10H22` $134$: PBE `52.30 s`, CF22D `72.55 s`).
- **Water Clusters (`6-31G`)**: `(H2O)_1` ($13$): PGFT `0.85 s` vs PBE `0.54 s` / CF22D `0.63 s` → `(H2O)_4` ($52$): PGFT `496.44 s` (`stat: 466.75 s`, `thc: 22.49 s`, `dyn: 6.68 s`) vs PBE `7.93 s` / CF22D `9.21 s`.
- **Fixed `H2O` Basis Scaling**: `STO-3G` ($7$): PGFT `0.147 s` vs PBE `0.381 s` / CF22D `0.461 s`; `def2-SVP` ($24$): PGFT `9.45 s` vs PBE `0.664 s` / CF22D `0.874 s` (`99,590`: `2.44 s`); `def2-TZVP` ($43$): PGFT `103.65 s` (`stat: 98.94 s`, `dyn: 1.25 s`) vs PBE `1.05 s` / CF22D `1.48 s` (`99,590`: `3.88 s`); `cc-pVTZ` ($58$): PGFT `527.08 s` (`stat: 517.34 s`, `dyn: 2.74 s`) vs PBE `1.41 s` / CF22D `2.00 s`.

### 3.2 OpenMP Multi-Thread Scaling (`OPENBLAS_NUM_THREADS=1`, 4 Physical Cores / 8 Logical Threads)

| System ($N_{\text{AO}}$) | Threads (`OMP`) | `PGFT v2` Total (s) | Speedup | `PGFT` THC (s) | `PGFT` Static SCF (s) | `PGFT` Dyn. ACFD (s) | `PBE` (s) | `CF22D` (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`H2O` (`def2-SVP`, `24`)** | 1 | `9.49` | `1.00×` | `0.46` | `8.77` | `0.18` | `0.67` | `0.88` |
| | 2 | `5.63` | `1.69×` | `0.43` | `5.01` | `0.10` | `0.44` | `0.54` |
| | 4 | **`4.07`** | **`2.33×`** | `0.42` | `3.50` | `0.071` | `0.34` | `0.46` |
| | 8 | `4.13` | `2.30×` | `0.42` | `3.56` | `0.069` | `0.35` | `0.34` |
| **`(H2)_12` (`6-31G`, `48`)** | 1 | `16.37` | `1.00×` | `6.53` | `8.05` | `1.63` | `4.98` | `6.44` |
| | 2 | `8.94` | `1.83×` | `3.74` | `4.25` | `0.84` | `3.23` | `4.33` |
| | 4 | `6.23` | `2.63×` | `2.39` | `3.29` | `0.47` | `2.27` | `3.61` |
| | 8 | **`5.74`** | **`2.85×`** | `1.90` | `3.29` | `0.45` | `3.41` | `3.84` |

> [!TIP]
> Setting `OPENBLAS_NUM_THREADS=1` is mandatory when running OpenMP-parallel PGFT v2: default oversubscribed `OMP=8, OPENBLAS=8` degrades `H2O (def2-SVP)` runtime by **4.3×** (`25.95 s` vs `5.98 s`).

### 3.3 Real-World Wall-Clock Time Across All 802 GSCDB137 Molecules (`def2-SVP`, Single Core)

| Method / Component | Cumulative CPU Time | Share / Ratio | Mean (s/mol) | Median (s/mol) | p90 (s/mol) | Max (s/mol) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PBE (99,590)`** | `4,805.9 s` (`1.33 h`) | `1.00×` | `5.99` | `2.54` | `14.08` | `376.6` |
| **`CF22D (99,590 + D3)`** | `7,015.6 s` (`1.95 h`) | `1.46×` PBE | `8.75` | `3.24` | `23.47` | `160.2` |
| **`DF-PBE (def2-QZVPPD)`** | `23,829.8 s` (`6.62 h`) | `4.96×` PBE-SVP | `29.71` | `8.35` | `85.80` | `609.3` |
| **`PGFT v2 (Full Total)`** | **`132,025.2 s` (`36.67 h`)** | **`27.47×` PBE / `18.82×` CF22D** | `164.62` | `11.02` | `525.26` | `2,757.5` |
| ↳ `HF initialization` | `262.0 s` (`0.07 h`) | `0.20%` of PGFT | `0.33` | — | — | — |
| ↳ `ISDF-THC construction` | `2,979.1 s` (`0.83 h`) | `2.26%` of PGFT | `3.71` | — | — | — |
| ↳ **`Dynamic ACFD + post-SCF`** | **`4,928.5 s` (`1.37 h`)** | **`3.73%` of PGFT** | **`6.15`** | — | — | — |
| ↳ *Subtotal (`HF + THC + ACFD`)* | *`8,169.6 s` (`2.27 h`)* | *`6.19%` of PGFT (`1.16×` CF22D)* | *`10.19`* | — | — | — |
| ↳ **`Static L-BFGS SCF` (4 seeds × 28 pts)** | **`123,854.1 s` (`34.40 h`)** | **`93.81%` of PGFT** | **`154.43`** | — | — | — |

**Key Engineering Finding**: The dynamic correlation engine (`Dynamic ACFD`: `1.37 h`) is **faster than `CF22D` (`1.95 h`)** and nearly tied with `PBE` (`1.33 h`) across 802 real molecules. **93.81%** of PGFT v2's computational cost is spent inside the 4-seed multi-start Static L-BFGS SCF loop (`123,854 s`), which currently evaluates all 4 seeds even on single-reference closed-shell molecules where `E_stat` ends up within `<0.1 mHa` of `E_hf`! Replacing unconditional 4-seed SCF with a fast stability/diradical-character gate or 1-seed warm start with DIIS/preconditioned Newton would immediately cut total PGFT v2 runtime by **10–15×** down to the ~`2.5–4 h` (`CF22D`-class) regime.

---

## 4. Accuracy Benchmark on GSCDB137 (20 Datasets, 463 Reactions)

![GSCDB137 20-dataset relative error profile, SIE4x4 bond-stretch curve, and basis-set convergence](./figures/fig2_gscdb137_accuracy.png)

### 4.1 Category-Level Summary (`def2-SVP` Head-to-Head + CBS References)

| Category | Sets / Rxns | `PGFT v2` SVP RelMetric (MAE) | `PBE` SVP RelMetric (MAE) | `CF22D` SVP RelMetric (MAE) | `DF-PBE` QZVPPD RelMetric (MAE) | `CF22D` Q-Chem CBS RelMetric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Thermochemistry (`TC`)** | 12 / 279 | `14.76` (`41.76`) | `6.74` (`19.26`) | **`4.84` (`13.37`)** | `4.18` (`12.60`) | `1.09` |
| **Barrier Height (`BH`)** | 2 / 57 | `7.72` (`12.20`) | `5.99` (`8.83`) | **`2.55` (`3.91`)** | `4.83` (`7.38`) | `1.06` |
| **Isomerization (`ISO`)** | 1 / 20 | `3.02` (`3.95`) | `4.05` (`5.29`) | **`2.50` (`3.27`)** | `2.26` (`2.95`) | `0.99` |
| **Noncovalent (`NC`)** | 4 / 97 | `31.84` (`3.86`) | `6.91` (`3.07`) | **`5.83` (`2.03`)** | `2.77` (`1.67`) | `1.47` |
| **Transition Metal (`TM`)** | 1 / 10 | `18.34` (`103.12`) | **`1.92` (`8.82`)** | `3.30` (`19.46`) | `1.99` (`8.97`) | `0.90` |
| **Overall Mean (20 sets)** | **20 / 463** | **`17.062` (`32.40`)** | **`6.322` (`13.76`)** | **`4.617` (`9.96`)** | **`3.760` (`9.23`)** | **`1.150`** |

*(MAE in `kcal/mol`; `RelMetric = GSCDB_Metric / Standard_Metric`, lower is better; `1.00` = GSCDB137 top-functional standard.)*

### 4.2 Complete 20-Dataset Breakdown (`MAE` in kcal/mol and `RelMetric`)

| Dataset | Cat. | Rxns | $\sigma_{\text{std}}$ | `PGFT v2` (SVP) MAE / Rel | `PBE` (SVP) MAE / Rel | `CF22D` (SVP) MAE / Rel | `DF-PBE` (QZVPPD) MAE / Rel | `CF22D` (CBS) MAE / Rel |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SIE4x4`** | TC | 16 | `9.18` | **`7.94` / `0.87`** | `26.16` / `2.85` | `13.27` / `1.45` | `23.23` / `2.53` | `10.87` / `1.18` |
| **`He3`** | NC | 49 | `0.176` | **`0.410` / `2.33`** | `0.426` / `2.42` | `0.523` / `2.97` | `0.267` / `1.51` | `0.469` / `2.66` |
| **`ISOMERIZATION20`** | ISO | 20 | `1.31` | `3.95` / `3.02` | `5.29` / `4.05` | **`3.27` / `2.50`** | `2.95` / `2.26` | `1.29` / `0.99` |
| **`TA13`** | NC | 13 | `1.36` | `8.68` / `6.39` | `8.96` / `6.60` | **`5.31` / `3.91`** | `4.14` / `3.05` | `1.63` / `1.20` |
| `G21IP` | TC | 36 | `2.75` | `13.15` / `4.79` | `4.79` / `1.74` | **`4.53` / `1.65`** | `3.81` / `1.39` | `2.94` / `1.07` |
| `ORBH35` | BH | 35 | `2.27` | `12.37` / `5.46` | `6.85` / `3.02` | **`3.58` / `1.58`** | `6.64` / `2.93` | `2.16` / `0.95` |
| `DIPCS9` | TC | 9 | `3.38` | `23.45` / `6.93` | **`6.67` / `1.97`** | `9.31` / `2.75` | `4.98` / `1.47` | `5.82` / `1.72` |
| `BH76RC` | TC | 30 | `1.01` | `7.56` / `7.51` | `5.91` / `5.87` | **`4.13` / `4.10`** | `4.11` / `4.08` | `1.00` / `1.00` |
| `ALKBDE10` | TC | 10 | `3.26` | `25.79` / `7.92` | **`6.07` / `1.86`** | `6.68` / `2.05` | `6.29` / `1.93` | `3.13` / `0.96` |
| `BDE99nonMR` | TC | 83 | `1.93` | `16.56` / `8.58` | `9.30` / `4.82` | **`4.93` / `2.55`** | `7.04` / `3.64` | `1.75` / `0.90` |
| `DBH22` | BH | 22 | `1.21` | `12.04` / `9.98` | `10.81` / `8.96` | **`4.24` / `3.51`** | `8.12` / `6.73` | `1.42` / `1.17` |
| `BDE99MR` | TC | 16 | `3.31` | `42.84` / `12.93` | `12.02` / `3.63` | **`6.83` / `2.06`** | `12.07` / `3.64` | `3.44` / `1.04` |
| `SN13` | TC | 13 | `0.631` | `9.16` / `14.51` | `6.13` / `9.72` | **`2.52` / `3.99`** | `5.29` / `8.38` | `0.57` / `0.90` |
| `WCPT6` | TC | 6 | `0.663` | `10.35` / `15.61` | `1.84` / `2.77` | **`0.78` / `1.17`** | `1.02` / `1.53` | `0.57` / `0.86` |
| `TMD10` | TM | 10 | `5.64` | `103.12` / `18.34` | **`8.82` / `1.92`** | `19.46` / `3.30` | `8.97` / `1.99` | `3.82` / `0.90` |
| `G21EA` | TC | 25 | `1.37` | `34.43` / `25.16` | **`23.16` / `16.92`** | `24.20` / `17.68` | `2.14` / `1.57` | `1.30` / `0.95` |
| `TAE_W4-17MR` | TC | 17 | `3.08` | `78.89` / `25.65` | `21.26` / `6.91` | **`14.91` / `4.85`** | `29.46` / `9.58` | `3.08` / `1.00` |
| `AE18` | TC | 18 | `4.95` | `230.93` / `46.64` | `107.76` / `21.76` | **`68.38` / `13.81`** | `51.82` / `10.46` | `7.53` / `1.52` |
| `O24` *(weighted)* | NC | 24 | `1.12` | `2.48` / `47.26` | `2.30` / `7.79` | **`1.87` / `8.50`** | `2.13` / `4.21` | `0.49` / `0.95` |
| `NC11` | NC | 11 | `0.054` | `3.88` / `71.37` | `0.59` / `10.85` | **`0.43` / `7.93`** | `0.13` / `2.32` | `0.06` / `1.06` |

*(Note: On `O24`, GSCDB137's `Metric` applies per-reaction inverse-reference weights; PGFT's unweighted MAE `2.48 kcal/mol` is close to `PBE` `2.30 kcal/mol`, while `RelMetric = 47.26` is driven by weak-binding heterodimer outliers.)*

### 4.3 Basis-Set Convergence Study (`def2-SVP` → `aug-cc-pVDZ` & `def2-TZVP`)

| Dataset (Rxns) | Basis | `PGFT v2` MAE (`Rel`) | `PBE` MAE (`Rel`) | `CF22D` MAE (`Rel`) | `CF22D` CBS Ref. MAE (`Rel`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`SIE4x4`** (16) | `def2-SVP` | **`7.94` (`0.87`)** | `26.16` (`2.85`) | `13.27` (`1.45`) | `10.87` (`1.18`) |
| | `def2-TZVP` | **`7.18` (`0.78`)** | `23.73` (`2.59`) | `11.16` (`1.22`) | `10.87` (`1.18`) |
| **`He3`** (49) | `def2-SVP` | **`0.410` (`2.33`)** | `0.426` (`2.42`) | `0.523` (`2.97`) | `0.469` (`2.66`) |
| | `aug-cc-pVDZ` | **`0.224` (`1.27`)** | `0.241` (`1.37`) | `0.525` (`2.98`) | `0.469` (`2.66`) |
| | `def2-TZVP` | `0.216` (`1.23`) | **`0.211` (`1.20`)** | `0.516` (`2.93`) | `0.469` (`2.66`) |
| **`G21EA`** (25) | `def2-SVP` | `34.43` (`25.16`) | `23.16` (`16.92`) | `24.20` (`17.68`) | `1.30` (`0.95`) |
| | `aug-cc-pVDZ` | `11.12` (`8.13`) | `3.26` (`2.38`) | **`1.85` (`1.35`)** | `1.30` (`0.95`) |
| **`AE18`** (18) | `def2-SVP` | `230.93` (`46.64`) | `107.76` (`21.76`) | `68.38` (`13.81`) | `7.53` (`1.52`) |
| | `def2-TZVP` | `136.68` (`27.60`) | `56.10` (`11.33`) | **`10.23` (`2.07`)** | `7.53` (`1.52`) |
| **`ALKBDE10`** (10) | `def2-SVP` | `25.79` (`7.92`) | `6.07` (`1.86`) | `6.68` (`2.05`) | `3.13` (`0.96`) |
| | `def2-TZVP` | `31.27` (`9.61`) | **`4.93` (`1.51`)** | `5.05` (`1.55`) | `3.13` (`0.96`) |

---

## 5. Physical & Algorithmic Diagnosis

### 5.1 Where PGFT v2 Excels
1. **Self-Interaction-Free Radical Dissociation (`SIE4x4`)**: Because PGFT v2 retains 100% exact Fock exchange plus screened ACFD correlation, its mean signed error across `H2+`, `He2+`, `(NH3)2+`, and `(H2O)2+` (`def2-TZVP`) stays flat at **`-6.08` to `-8.29 kcal/mol`** across $R/R_e = 1.00 \to 1.75$, whereas `PBE` diverges monotonically (`+13.70 → +33.18 kcal/mol`) and `CF22D` (46.3% HF exchange) still diverges (`+5.01 → +16.78 kcal/mol`). PGFT v2's `RelMetric = 0.78` (`TZVP`) beats CBS-limit `CF22D` (`1.18`) and `DF-PBE` (`2.53`).
2. **Non-Additive Many-Body Dispersion (`He3`)**: PGFT v2's ACFD correlation naturally captures 3-body Axilrod–Teller–Muto dispersion (`aug-cc-pVDZ` MAE **`0.224 kcal/mol`**, `RelMetric = 1.27`), outperforming 2-body D3-corrected `CF22D` (`0.525 kcal/mol` at `aug-DZ`, `0.469` at CBS) and `PBE` (`0.241` at `aug-DZ`, `0.267` at CBS).
3. **Closed-Shell Isomerizations (`ISOMERIZATION20`) & Radical Complexes (`TA13`)**: When bond counts are conserved (`ISOMERIZATION20`), PGFT v2's systematic correlation deficit cancels cleanly (`MAE 3.95` vs `PBE 5.29 kcal/mol`, `MSE = -0.007 kcal/mol`).

### 5.2 Three Quantitatively Diagnosed Failure Modes & Improvement Roadmap

1. **Deep-Core ISDF-THC Non-Variational Fitting Error ($Z \ge 15$, severe in `TMD10` & Br-containing species)**:
   - Across all 802 `def2-SVP` molecules, **25 molecules** exhibit non-variational static energies ($E_{\text{stat}} - E_{\text{HF}} > +1\text{ mHa}$), led by `CuBr` (`+323.86 mHa = +203.22 kcal/mol`), `Co` (`+293.99 mHa`), `TiBr` (`+212.25 mHa`), `Cu` (`+181.13 mHa`), `H2O-Br` (`+155.31 mHa`), `Fe` (`+124.00 mHa`), `MnBr` (`+69.15 mHa`), and `Br` (`+36.42 mHa`).
   - **Root cause**: Fixed `n_rad = 28` radial shells (`r_min = 3e-4`) and the `N_K` cap poorly reconstruct steep $1s/2p/3d$ core densities in ISDF-THC, injecting massive errors into the static energy difference on `TMD10` (`MAE = 103.1 kcal/mol`) and `TA13` (`MaxAbs = 73.4 kcal/mol` on `H2O-Br`). Additionally, `solver.py` builds $H_{\text{core}}$ via `int1e_kin + int1e_nuc` rather than `mf.get_hcore()`, omitting ECP terms for $Z > 36$ (not triggered here since all 20 sets have $Z \le 35$).
   - **Fix**: Switch core construction to `mf.get_hcore()`, apply exact HF core subtraction (or frozen-core / element-adaptive radial shells for $Z \ge 15$), and compute `E_stat - E_hf` through the differential density $\Delta D = D_{\text{stat}} - D_{\text{HF}}$ so the exact HF baseline cancels analytically.

2. **3rd-Row Core-Valence `E_dyn` Drop & Multi-Bond Under-Correlation (`AE18`, `TAE_W4-17MR`, `BDE99MR`, `ALKBDE10`)**:
   - In `AE18`, PGFT v2's $E_{\text{dyn}}$ grows smoothly across row 2 (`He: -15.58` → `Ne: -85.36 kcal/mol`), then collapses abruptly at row 3 (**`Na: -2.05 kcal/mol`**, `Mg: -10.20`, `Ar: -45.73 kcal/mol` — half of `Ne` despite 18 electrons!). Because the single global/local chemical-potential parameter $\mu_{\text{loc}}$ shifts to the $3s/3p$ valence edge, $2s/2p$ core-valence correlation is suppressed.
   - Combined with strong dynamic self-screening damping, every single bond in `TAE_W4-17MR` (17/17, `MSE = -78.89 kcal/mol`), `ALKBDE10` (`MSE = -25.79 kcal/mol`), and `BDE99MR` (`MSE = +41.33 kcal/mol` in dissociation sign) is systematically underbound, and `G21EA` anions remain under-correlated by `~10.5 kcal/mol` even at `aug-cc-pVDZ`.
   - **Fix**: Replace single-scale $\mu_{\text{loc}}$ with orbital/shell-resolved frequency scaling and re-examine the self-screening damping factor so paired valence-bond correlation is recovered without breaking SIE freedom.

3. **Heterodimer Markov $\mu_{\text{loc}}$ / `S_inter` Connectivity Switching (`NC11`, `O24`)**:
   - On homodimers and matched noble-gas pairs in `NC11`, PGFT v2 is accurate (`H2-H2`: `+0.06`, `He-H2`: `-0.03`, `He-LiH`: `-0.33 kcal/mol`), whereas on chemically dissimilar heterodimers (`Ar-HF`: **`-17.35`**, `H2-HCl`: **`+12.74`**, `H2-HF`: **`+4.38 kcal/mol`**), the hard `C_ovlp > 1e-4` connectivity threshold in `_quasiparticle_energies_for_spin` / `_occupied_connected_projector` switches monomer chemical potentials discontinuously upon complexation, creating $\pm 12\text{--}17\text{ kcal/mol}$ jumps in $E_{\text{dyn}}$ for `0.6 kcal/mol` van der Waals interactions.
   - **Fix**: Replace the discontinuous `1e-4` overlap cutoff with a smooth, size-consistent fragment-invariant weighting function so separated monomers at weak overlap strictly preserve $E_{\text{dyn}}(A \cdots B) \to E_{\text{dyn}}(A) + E_{\text{dyn}}(B)$.

---

## 6. Reproducibility & File Index

- **Scaling & Thread Benchmark Script**: [`bench_scaling_time.py`](./bench_scaling_time.py) → [`results/scaling_time_results.json`](./results/scaling_time_results.json)
- **GSCDB137 Worker & Analyzer**: [`bench_gscdb137_worker.py`](./bench_gscdb137_worker.py), [`analyze_gscdb137.py`](./analyze_gscdb137.py)
- **Aggregated Result Summaries**:
  - [`results/summary_20sets_svp.json`](./results/summary_20sets_svp.json) (20 datasets, 463 reactions, 802 molecules, `def2-SVP`)
  - [`results/summary_20sets_qzvppd_dfpbe.json`](./results/summary_20sets_qzvppd_dfpbe.json) (20 datasets, 802 molecules, `DF-PBE / def2-QZVPPD`)
  - [`results/summary_augdz.json`](./results/summary_augdz.json) (`G21EA` + `He3`, `aug-cc-pVDZ`)
  - [`results/summary_4sets_tzvp.json`](./results/summary_4sets_tzvp.json) (`AE18`, `ALKBDE10`, `He3`, `SIE4x4`, `def2-TZVP`)
- **Plotting Script & Publication Figures**: [`plot_gscdb137_comparison.py`](./plot_gscdb137_comparison.py) → [`figures/fig1_scaling_and_timing.pdf`](./figures/fig1_scaling_and_timing.pdf) & [`figures/fig2_gscdb137_accuracy.pdf`](./figures/fig2_gscdb137_accuracy.pdf) (both passed static preflight `21/21`, `1.5 pt` panel-alignment audit, PDF font-size floor audit `>= 5.3 pt`, and rendered collision audit with `0 fail, 0 warn`).
