# Reviewer #2 Mathematical & Algorithmic Audit Checklist (`referee_checklist.md`)

**Role**: Uncompromising Referee #2 (*Nature*, *Science*, *Physical Review X*)  
**Scope**: Audit of all proposed non-DFT $\mathcal{O}(N^3)$ electronic structure theories for universal static + dynamic correlation.

Any submitted theory (`theory_A`, `theory_B`, or unified synthesis) must survive **every** mathematical, algorithmic, and numerical gate below. Failure on any gate triggers an immediate **Major Revision / Reject** report with explicit counterexamples.

---

## Gate 1: Strict Non-DFT Purity (Zero Smuggled Density Functionals)

We audit both the mathematical equations and the Python AST/imports of the implementation:

1. **Forbidden Hohenberg-Kohn / Kohn-Sham Ingredients**:
   - Zero calls to `pyscf.dft`, `RKS`, `UKS`, `GKS`, `dft.Grids`, `libxc`, `xcfun`, or numerical real-space Becke quadrature grids for exchange-correlation energy densities $e_{xc}[\rho(\mathbf{r}), \nabla\rho(\mathbf{r}), \tau(\mathbf{r})]$.
   - Zero semi-local exchange-correlation kernels $f_{xc}[\rho](\mathbf{r}, \mathbf{r}') = \delta^2 E_{xc}[\rho]/\delta\rho(\mathbf{r})\delta\rho(\mathbf{r}')$ smuggled inside RPA/TDHF/BSE propagators (e.g., ALDA, range-separated short-range DFT hybrids).
   - Zero Kohn-Sham starting orbitals (`mf = dft.RKS(mol)`). The theory must construct its orbitals self-consistently from its own wave-function / Green's-function / RDM / fluctuation functional, or start solely from pure ab initio wave-function references (e.g., RHF/ROHF or self-consistent natural orbitals).
2. **No Molecule-Specific Heuristics or Overfitted Parameters**:
   - Any `if mol.nelectron == ...` switches, atom-lookup tables, empirical dispersion patches (`DFT-D3`/`D4`), or parameters fitted to the benchmark suite constitute immediate disqualification. Any universal mathematical constants must be derived from exact limits (e.g., 1-electron exactness, uniform electron gas / 2-electron exactness, high-density Gell-Mann–Brueckner limit, or fluctuation-dissipation theorem) and frozen across the entire periodic table.

---

## Gate 2: Algebraic $\mathcal{O}(N^3)$ Complexity Audit (Line-by-Line Tensor Inspection)

Let $N$ denote system size, with $N_{\text{AO}}, N_{\text{MO}}, N_{\text{occ}}, N_{\text{vir}}, N_{\text{aux}}, N_{\text{THC}} \propto \mathcal{O}(N)$, while frequency/time/coupling quadrature counts $N_\omega, N_\tau, N_\lambda \sim \mathcal{O}(1)$ are strictly independent of $N$.

### 2.1 Tensor Contraction Index-Counting Rule
Every `np.einsum`, `scipy.linalg`, and matrix product is audited by counting unique $\mathcal{O}(N)$-scaling indices in each elementary pairwise contraction:
- **Allowed ($\le 3$ $\mathcal{O}(N)$ indices)**:
  - Matrix-matrix multiplications $(M \times K)(K \times L) \to \mathcal{O}(N^3)$.
  - Cholesky / eigendecomposition / LU / SVD / matrix logarithm / matrix square root of $N_{\text{MO}} \times N_{\text{MO}}$ or $N_{\text{aux}} \times N_{\text{aux}}$ matrices: $\mathcal{O}(N^3)$ per quadrature point, total $\mathcal{O}(N_\omega N^3) = \mathcal{O}(N^3)$.
  - Coulomb matrix via density fitting: $v_P = \sum_{\mu\nu} (P|\mu\nu) P_{\mu\nu}$ is $\mathcal{O}(N_{\text{aux}} N_{\text{AO}}^2) = \mathcal{O}(N^3)$.
- **FORBIDDEN ($\ge 4$ $\mathcal{O}(N)$ indices — Hidden $\mathcal{O}(N^4)$, $\mathcal{O}(N^5)$, or $\mathcal{O}(N^6)$ Traps!)**:
  1. **4-Center ERI Construction**: Calling `mol.intor('int2e')` or `ao2mo.kernel` to form $(pq|rs)$ scales as $\mathcal{O}(N^4)$ storage and $\mathcal{O}(N^5)$ transformation. **Instant rejection.**
  2. **Canonical MO DF Transformation**: Forming $B_{pq}^P = \sum_{\mu\nu} (P|\mu\nu) C_{\mu p} C_{\nu q}$ via $T_{\mu q}^P = \sum_\nu (P|\mu\nu) C_{\nu q}$ has **four** $\mathcal{O}(N)$ indices $(P, \mu, \nu, q) \implies \mathcal{O}(N_{\text{aux}} N_{\text{AO}}^2 N_{\text{MO}}) = \mathcal{O}(N^4)$! Even exact HF exchange $K_{\mu\nu} = \sum_{Pi} B_{\mu i}^P B_{\nu i}^P$ requires forming $B_{\mu i}^P$ in $\mathcal{O}(N_{\text{aux}} N_{\text{AO}}^2 N_{\text{occ}}) = \mathcal{O}(N^4)$ unless reduced via THC / ISDF ($B_{\mu\nu}^P = \sum_r X_{\mu r} X_{\nu r} Z_r^P$), localized support, or rank-factorized interpolative decomposition!
  3. **Canonical Frequency-Space RI-RPA Polarizability Trap**:
     $$\Pi_{PQ}(i\omega) = 2 \sum_{i}^{\text{occ}} \sum_{a}^{\text{vir}} B_{ia}^P B_{ia}^Q \frac{2\epsilon_{ia}}{\omega^2 + \epsilon_{ia}^2}$$
     Contraction `einsum('Pia,Qia->PQ', ...)` touches **four** $\mathcal{O}(N)$ indices $(P, Q, i, a)$! Its cost is $\mathcal{O}(N_\omega N_{\text{aux}}^2 N_{\text{occ}} N_{\text{vir}}) = \mathcal{O}(N^4)$, **NOT** $\mathcal{O}(N^3)$! True $\mathcal{O}(N^3)$ polarizability requires THC / ISDF / separable real-space or interpolative factors where $\chi_{rr'}(i\tau) = -2 G_{rr'}(i\tau) G_{r'r}(-i\tau)$ is $\mathcal{O}(N^2)$ and projection $\sum_{rr'} Z_r^P \chi_{rr'}(i\tau) Z_{r'}^Q$ is $\mathcal{O}(N^3)$.
  4. **SOSEX / Screened Exchange / Vertex Contraction Trap**: Any contraction over $(P, Q, i, j, a, b)$ or $(P, p, q, r)$ scales as $\mathcal{O}(N^4)-\mathcal{O}(N^5)$ unless factorized through $\mathcal{O}(N)$ THC/interpolative intermediates with strictly 3-index contractions.
  5. **Particle-Hole / Particle-Particle Casida Diagonalization Trap**: Diagonalizing an $N_{\text{occ}}N_{\text{vir}} \times N_{\text{occ}}N_{\text{vir}}$ RPA/BSE/pp-RPA matrix costs $\mathcal{O}(N^6)$!
  6. **Quadrature Grid Inflation**: Claiming $\mathcal{O}(N_\omega N^3)$ while letting $N_\omega \propto N$ (or hiding a dense 3D spatial grid of $5000 \cdot N_{\text{atom}}$ points without low-rank compression).

### 2.2 Empirical Asymptotic Scaling Verification
We fit $\log(\text{FLOPs or wall time})$ vs $\log(N)$ across linear $\text{H}_{2k}$ chains ($k = 4, 8, 12, 16, 24, 32$) with fixed quadrature settings and verify the empirical exponent $\alpha \le 3.1$.

---

## Gate 3: Size-Consistency & Size-Extensivity Torture Tests

Many "clever" multireference-regularized theories pass single-molecule dissociation curves by cheating on size-consistency! We test four distinct regimes at $R_{\text{sep}} = 100\text{ \AA}$ (and $\text{H}_{2k}$ chains):

1. **Test A — Closed-Shell Monomers**:
   $$\Delta E_{\text{SC,A}} = E(\text{He}\cdots\text{He}, 100\text{ \AA}) - 2 E(\text{He}) = 0 \quad (|\Delta E| < 10^{-7}\text{ Ha})$$
2. **Test B — Two Strongly Correlated Stretched Singlets (1-RDM / Global Entropy Killer)**:
   $$\Delta E_{\text{SC,B}} = E\big(\text{H}_2(3.2\text{ \AA}) \cdots \text{H}_2(3.2\text{ \AA}), 100\text{ \AA}\big) - 2 E\big(\text{H}_2(3.2\text{ \AA})\big) = 0$$
   - *Landmine*: Any theory that uses a global nonlinear function of total entanglement entropy $S = -\text{Tr}[\gamma \ln \gamma + (1-\gamma)\ln(1-\gamma)]$, global purity $\text{Tr}(\gamma - \gamma^2)$, global Casimir invariant, or Brillouin-Wigner denominator shift $\Delta - E_{\text{corr}}$ **fails Test B** by tens of mHa!
3. **Test C — Mixed Static + Dynamic Supersystem (Global Gap-Switch Assassin)**:
   $$\Delta E_{\text{SC,C}} = E\big(\text{H}_2(3.2\text{ \AA}) \cdots \text{He}, 100\text{ \AA}\big) - E\big(\text{H}_2(3.2\text{ \AA})\big) - E(\text{He}) = 0$$
   - *Landmine*: Any theory that senses multireference character via the global HOMO-LUMO gap $\Delta \epsilon_{\text{HOMO-LUMO}}$ of the supersystem sees $\Delta \epsilon_{\text{global}} = \Delta \epsilon_{\text{H}_2} \approx 0$ and globally damps or switches off dynamic correlation — **erasing the correlation energy of the spectator $\text{He}$ atom $100\text{ \AA}$ away!**
4. **Test D — Thermodynamic Extensivity on $\text{H}_{2k}$ Chains ($k = 1, 2, 3, 4, 5$)**:
   - Checks that $E(\text{H}_{2k}) / k$ converges linearly to a finite thermodynamic limit without $\propto \sqrt{k}$ Brillouin-Wigner sub-extensivity or CEPA-0 divergence.
5. **Heteronuclear Dissociation Limit ($\text{LiH}$, $\text{HF}$)**:
   - Checks proper neutral-atom dissociation without spurious fractional charges (which plague semilocal DFT and unscreened RPA due to convex/concave $E(N)$ errors).

---

## Gate 4: Strict Spin Purity ($\langle \hat{S}^2 \rangle = S(S+1)$)

1. **No Secret UHF Symmetry Breaking**:
   - Closed-shell singlets ($\text{H}_2$, $\text{H}_4$, $\text{H}_2\text{O}$, $\text{N}_2$, $\text{F}_2$, $\text{LiH}$, $\text{HF}$) must maintain spin-restricted 1-RDM symmetry $\gamma^\alpha = \gamma^\beta$ and spin purity $\langle \hat{S}^2 \rangle = 0$ at all bond lengths $R \in [0.5, 10.0]\text{ \AA}$.
   - Achieving flat dissociation by breaking $\alpha/\beta$ spin symmetry (UHF/UKS Coulson-Fischer kink + $\langle \hat{S}^2 \rangle = 1, 2, 3$ spin contamination) is **strictly disqualified**.
2. **Triplet-Instability Resilience**:
   - Beyond the Coulson-Fischer point, the RHF Hessian $\mathbf{A}+\mathbf{B}$ or triplet stability matrix develops **negative eigenvalues**. Any RPAX / BSE / PhPA / TDHF theory that evaluates $(\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{A}+\mathbf{B})(\mathbf{A}-\mathbf{B})^{1/2}$ without curing triplet instabilities produces **complex (`NaN`) energies** or non-analytic cusps (`max(0, eig)`). We explicitly test across the Coulson-Fischer transition and $\text{H}_4$ $D_{4h}$ degeneracy!

---

## Gate 5: One-Electron & Two-Electron Exactness Limits

1. **One-Electron Self-Interaction Freedom (`cc-pVDZ`)**:
   - Exact correlation energy for any $N_e = 1$ system MUST vanish identically:
     $$E_{\text{corr}}(\text{H}) = 0, \qquad E_{\text{corr}}\big(\text{H}_2^+(R)\big) = 0 \quad \forall R \in [0.5, 10.0]\text{ \AA}$$
   - *Landmine*: Direct RPA (dRPA) gives $E_{\text{corr}}(\text{H}) \approx -20\text{ mHa}$ and catastrophically overbinds stretched $\text{H}_2^+$ by $> 50\text{ kcal/mol}$! Furthermore, a naive `if mol.nelectron == 1: return E_hf` check is **cheating** — we test $\text{H}_2^+(R=6\text{ \AA}) \cdots \text{He}(100\text{ \AA})$ (a 3-electron supersystem!) to catch hard-coded `nelectron == 1` hacks!
2. **Two-Electron Dynamic vs Static Correlation (`cc-pVDZ`)**:
   - Pure dynamic correlation across nuclear charge scaling: $\text{H}^-$ ($Z=1$, diffuse strongly correlated anion), $\text{He}$ ($Z=2$), $\text{Li}^+$ ($Z=3$, high-density limit).
   - Smooth transition from dynamic correlation ($\text{H}_2$ at $R = 0.5 - 0.74\text{ \AA}$) to pure static entanglement ($\text{H}_2$ at $R = 3.2 - 4.5\text{ \AA}$) without an unphysical Coulson-Fischer bump (which plagues dRPA and RPA+SOSEX at intermediate $R \approx 1.8 - 2.5\text{ \AA}$!).

---

## Gate 6: Multireference Bond Breaking & Diradical Smoothness (Zero Cusps, Zero Turnovers)

1. **Degenerate Diradical $\text{H}_4$ on a Circle ($\theta \in [70^\circ, 110^\circ]$)**:
   - At $\theta = 90^\circ$ ($D_{4h}$ square), canonical RHF suffers an exact HOMO-LUMO degeneracy ($b_{2u}$ vs $b_{3u}$) and exhibits a sharp derivative discontinuity $dE_{\text{RHF}}/d\theta|_{\theta=90^\circ^+} \neq dE_{\text{RHF}}/d\theta|_{\theta=90^\circ^-}$.
   - MP2 diverges to $-\infty$; R-CCSD(T) exhibits a violent downward cusp/inversion; even R-CCSD exhibits a noticeable cusp!
   - **Audit Test**: Compute finite-difference first and second derivatives around $\theta = 89.5^\circ, 90.0^\circ, 90.5^\circ$. Does the proposed theory remove the RHF orbital-switching cusp, or is it enslaved to canonical RHF occupation discontinuities?
2. **Simultaneous Multi-Bond Breaking**:
   - **Single bonds**: $\text{H}_2$, $\text{LiH}$, $\text{HF}$ ($R = 0.8 \to 3.6\text{ \AA}$) — check dissociation energy $D_e$ and asymptotic flatness.
   - **Symmetric double bond (4 entangled electrons)**: $\text{H}_2\text{O}$ symmetric stretch ($R_{\text{OH}} = 0.8 \to 3.0\text{ \AA}$) vs Exact Full FCI (`6-31G`).
   - **Triple bond (6 entangled electrons — The Ultimate Boss)**: $\text{N}_2$ dissociation ($R = 0.9 \to 3.2\text{ \AA}$) vs Exact Full FCI (`STO-3G`) and Frozen-1s Exact FCI (`6-31G`). Check explicitly for R-CCSD / R-CCSD(T)-style unphysical barrier + turnover to $-\infty$ at $R > 1.8\text{ \AA}$, and check that $\kappa$-regularized methods do NOT collapse back to the $+300\text{ kcal/mol}$ RHF ionic asymptote!

---

## Gate 7: Single-Reference Equilibrium Accuracy & Double-Counting Audit

A method that fixes $\text{N}_2$ dissociation by wrecking single-reference thermochemistry is useless. Without fitting parameters, we audit against Exact FCI / R-CCSD(T):

1. **No Static-Dynamic Double Counting**:
   - Check absolute correlation energies and equilibrium bond lengths $R_e$ of $\text{H}_2$, $\text{HF}$, $\text{H}_2\text{O}$, $\text{N}_2$. Double-counted theories systematically over-correlate equilibrium molecules by $30-150\text{ mHa}$ and over-elongate bonds.
2. **Reaction Energies**:
   - $\text{H}_2 + \text{F}_2 \to 2\text{HF}$
   - $\text{N}_2 + 3\text{H}_2 \to 2\text{NH}_3$
   - $\text{CO} + \text{H}_2\text{O} \to \text{CO}_2 + \text{H}_2$
   - $\text{CH}_4 + 2\text{H}_2\text{O} \to \text{CO}_2 + 4\text{H}_2$
   - $\text{Li}_2 + \text{H}_2 \to 2\text{LiH}$
3. **Conformational, Inversion, Radical, & Diradical Barrier Heights**:
   - $\text{H}_2\text{O}$ linear inversion barrier ($C_{2v} \to D_{\infty h}$)
   - $\text{NH}_3$ planar inversion barrier ($C_{3v} \to D_{3h}$)
   - $\text{C}_2\text{H}_6$ internal rotation barrier (staggered $D_{3d} \to$ eclipsed $D_{3h}$)
   - $\text{H} + \text{H}_2 \to [\text{H}\cdots\text{H}\cdots\text{H}]^\ddagger \to \text{H}_2 + \text{H}$ collinear hydrogen exchange barrier (classic DFT delocalization failure!)
   - $\text{BeH}_2$ $C_{2v}$ insertion diradical barrier ($\text{Be} + \text{H}_2 \to \text{BeH}_2$, classic multireference symmetry-forbidden Woodward-Hoffmann barrier!)
4. **Weak / Dispersion Interactions**:
   - Check whether non-overlapping closed-shell fragments ($\text{He}_2$ van der Waals well around $R \approx 3.0\text{ \AA}$) exhibit physically attractive $-C_6/R^6$ London dispersion rather than purely repulsive Hartree-Fock / 1-RDM J-K behavior.
