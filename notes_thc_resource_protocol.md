# THC resource-estimation protocol: rules, observations, TODO

Written 2026-10-22. Reconstructed from the final tables, raw optimization logs, the Codex
history, and arXiv:2610.02916 (Sec. III, App. E, G). The paper text is brief, and the
actual numbers only roughly follow it; this file records what was really done.

## 1. Where the final numbers live

- Final table: central `THC_general/table_norm_summary.txt` (2026-09-29). It equals DGX
  `THC_general/table_norm_summary.txt` (09-19) plus the CdSe rows.
- How the values were obtained: DGX `THC_general/table_interpolated_values.txt` lists the
  raw points each value was interpolated from.
- Raw exchange (ov) data: DGX `THC_general/opt_ov_Xo_colnorm/` (final; `p0`, i.e. a fixed
  norm ratio, not scaled with nk). Older `opt_ov/` (no colnorm, `p2`) is NOT used.
- Raw direct data: DGX `THC_general/opt_full/` (fullscreen) and `opt_full_AO/` (fullbare).
- No script generated the tables; they were computed ad hoc in agent sessions.

## 2. Quantities

- `norm_loss` (optimize_X_ov/full output) = $\Vert X^o\Vert^2 \Vert X^v\Vert^2 \Vert W\Vert / N_k$,
  with $\Vert X^o\Vert$ = max column norm (`--Xo_colnorm`), $\Vert W\Vert$ = max over q of
  max |eigenvalue|. This is the subnormalization proxy.
- `rel_error`: relative Frobenius error against the reference THC (large c_ref).
- Direct (full) fits: judged by the error of the projection onto (ov|vo) ("ovvo rel_error"),
  and report the oooo and vvvv norms; oovv norm = sqrt(oooo * vvvv).
- Loss (optimize_X_common.py): log10(eps/1e-2 + 1) + cosh(eps/0.1) - 1 + alpha * norm_loss.

## 3. Rules

### 3.1 Accuracy target
- Physical criterion: MP2 error <= 0.005 eV per active electron (realistic BSE sector,
  called "2--3 mEh" in the table); the older sector is 0.1 mEh per cell.
- The calibration of MP2 error vs (ov|vo) rel_error was done on diamond only (paper Fig. 7).
  Paper says the target is 2e-2; the tables actually used 3e-2 (0.005 eV/e sector) and 4e-3
  (0.1 mEh sector; 8e-3 for MgO).
- Exchange is done first; it tells how much (ov|vo) error is allowed for the MP2 target.
  The same ovvo error is then required for the direct term's (ov|vo) projection.

### 3.2 Exchange (bare (ov|vo), optimize_X_ov.py, --Xo_colnorm)
- Fixed alpha (p0), scan alpha (typ. 0.003, 0.01, 0.03, 0.1, 0.3, 1.0).
- For each (system, nk): take the two alpha points whose rel_error brackets the target,
  interpolate norm_loss to the target. Diamond / MgO / BN / AlN used linear interpolation
  in rel_error; CdSe used log-log (log norm_loss = a log rel_error + b). Use ONE convention
  in new work (log-log preferred).
- Avoid long extrapolation: the 0.1 mEh nk=2 values (diamond 0.354, BN 0.294, AlN 0.883)
  are linear extrapolations far outside the data and are unreliable. Run more alpha instead.
- Choice of c: cost ~ c^1.5 * norm_loss at the target error. Pick a c that roughly
  minimizes this and that can reach the target at all nk (not optimized per nk).
  Example (diamond): at 4e-3, c=5 is cheaper at nk=4 but cannot reach 4e-3 at nk=6
  (min 4.6e-3), so c=8 was used; at 3e-2, c=5 is clearly cheaper (36 vs ~66 at nk=6).
- Extrapolation in nk: log norm = a log nk + b (CdSe, to nk=10).

### 3.3 Direct (full ERI, optimize_X_full.py)
- Pick c so that the ovvo-projected error is near/below the exchange target; run a single
  alpha (0.1) and use it directly (no interpolation in the old tables; CdSe was interpolated).
- c=4 typical (diamond, MgO, BN, AlN at 3e-2); CdSe needed c=5 (c=4 gives 6.7e-2).

### 3.4 Reference THC (c_ref), user-confirmed 2026-10-22
- Optimized fits are measured against the reference THC, not GDF, so the reference error
  must be negligible.
- c_ref candidates are "good" numbers: 12, 16, 20, 24, 30, 36, 48, ...
- Exchange (ov) reference: smallest candidate c with reference
  rel_error vs GDF <= 1e-3 at ALL nk (factor 1.5 for c<=16, 3.0 for c=20 as before).
  Old data: diamond/BN 12, AlN 16, MgO and heavier 20 (ref errors 1-2e-3; slightly loose).
- Direct (full) reference: --abs with factor 5e-5, i.e. reference rel_error ~5e-5;
  c_ref 24 (diamond/BN/AlN) or 36 (MgO and heavier). Do NOT relax this: the full ERI has
  extra issues, even though the final full fit errors are ~1e-2.
- Final fit errors for comparison: exchange target 3e-2 (0.005 eV/e) or 4e-3 (0.1 mEh);
  direct ovvo-projected 3-4e-2 at c=4/5, full-tensor 2-8e-2 at c=4/5.

### 3.5 Where to run (storage rule)
- DGX has limited storage: never store GDF on DGX.
- DFT/GDF/GW/screening and ISDF generation (generate_isdf_gdf.py, reference + initial
  guesses) run on central. Then rsync only the DFT pkl and ISDF*.chk (and whatever small
  files optimization needs) to DGX.
- THC optimization (optimize_X_ov/full.py) and THC-only checks (custom_mp2_thc.py) run on
  DGX GPUs. Anything that needs GDF (custom_mp2_df.py, bse_df.py) runs on central.

### 3.6 Other conventions in the old tables
- MgO/BN/AlN high-error (0.005 eV/e) values were NOT fitted: value = low-error value x
  (diamond high / diamond low) at the same nk. New systems should be fitted directly.

## 4. Observations (verified on diamond, MgO, BN, AlN, CdSe)

- Exchange: at fixed alpha, rel_error grows with nk (e.g. BN c=8 alpha=0.1: 1.6, 2.9, 4.5,
  5.8, 7.0 e-3 for nk=2..6; MgO saturates at nk>=5). Reason: the error term is log-like and
  ~Nk-independent while the norm needed for low error grows with Nk. Consequence: the
  bracketing alpha pair moves to smaller alpha as nk grows; every nk needs its own scan.
- Direct: at fixed c, the ovvo-projected error is ~nk-independent (within ~10%, nk=2 slightly
  higher), e.g. BN c=4: 3.8, 3.4, 3.5, 3.5, 3.4 e-2; AlN c=6: 0.56, 0.56, 0.60, 0.52 e-2.
  It depends weakly on alpha (diamond nk2 c6: alpha 0.03 -> 1.0 gives 3.6e-3 -> 7.0e-3).
  oooo/vvvv norms are also ~nk-independent. The full-tensor rel_error decreases with nk.
- MoSe2bulk results are invalid (see 5.2).

## 5. Code / data issues found 2026-10-21/22

### 5.1 2D signed GDF (central/local code, synced to central THC_general)
- 2D GDF (cell.dimension=2, low_dim_ft_type=None) has a negative auxiliary channel `j3c-`
  at q=0: (pq|rs) = sum_x S_x R_x,pq R_x,rs^*, S = +1/-1. The negative row is NOT small
  (black P 2x2x1: comparable to all positive rows; dominant in oo/vv, small in ov), likely
  the 2D truncated-Coulomb G=0 term (-2 pi R_c^2).
- utils.get_gdf_tensor_compact(..., signed=True) returns (R, S); S passed through
  df_inner_from_mo, thc_df_ov_rhs_from_mo and wrappers. generate_isdf_gdf.py always signed;
  custom_mp2_df.py signed (2D --exact uses PySCF ao2mo path); optimize_X_ov/full `--2d`.
- Verified on black P 2x2x1: ERI vs PySCF ao2mo ~1e-14; DF MP2 vs PySCF 6e-14 Ha;
  ISDF ov c=8 rel 2.7e-2, full c=6 rel 1.4e-1 (reference fits run fine).

### 5.2 CDERIArray truncation bug (affects old 3D data)
- PySCF CDERIArray.load truncates every block to the naux of the first block. The old
  loader used it. Fixed: loader now uses _load3c and zero-pads.
- Affected: MoSe2bulk (naux 784-794; dropped rows carried 28-86% of |L|^2) -> all MoSe2bulk
  ISDF results must be redone. All other 3D systems have uniform naux (unaffected).
  black P 2D q!=0 blocks (500 vs 474) were also affected before the fix.

### 5.3 Ill-conditioned j2c: Cholesky vs eigen-decomposition (2026-10-22)
- PySCF GDF tries Cholesky of j2c(q) first and only falls back to eigen-decomposition (with
  linear-dependence removal) if Cholesky fails. With the near-linearly-dependent ETB auxiliary
  basis, Cholesky "succeeds" at some q and amplifies noise into spurious large components.
- MoSe2D 3x3x1: symmetry-equivalent q=(0,1/3) [ED, 402 rows] and q=(1/3,0) [CD, 494 rows] had
  |V|^2 = 119 vs 300; ov ISDF errors 8% vs 76% (total 59%, refs useless for nk>=3).
  With eigen-decomposition at every q: all q 400-403 rows, symmetric norms, all q ~8%.
- The AO basis is fine (min S eigenvalue ~1e-2); switching to dzv is not needed.
- Fix: settings.pydata `"gdf": {"j2c_eig_always": True}` -> do_dft_gdf.py sets
  `_RSGDFBuilder.j2c_eig_always`. Only the GDF changes (PBE SCF uses q=0, which already went ED).
- Symptom to watch: non-uniform naux across q, or asymmetric per-q norms (MoSe2bulk had
  784-794 -> also affected; not needed now). Uniform-naux 3D systems may still have
  ill-conditioned CD at some q; not checked yet.
- MoSe2D rerun with the fix (old files in data dir `old_cd/`).

## 6. TODO (2D: blackP2D, MoSe2D, CdSe2D; 3D: PbSebulk, CdSezb)

- Data on central `THC_general/data_*_gth-dzvp-molopt-sr_final/`.
  2D: DFT+GDF done (blackP/MoSe2 up to 8x8x1, CdSe2D up to 4x4x1). N_k = n^2.
  PbSe bulk: workflow done 2-5 (gap at L; odd meshes miss L -> use even meshes for physics).
  CdSezb (zinc blende, 2 atoms): workflow 2-5 submitted (5309038-57); 2,3 done.
- Remaining code adaptation for 2D: run/submit scripts (run_isdf.sh has nk nk nk),
  load_isdf_ov/full.py kmesh, bse_df.py signed (optional), maybe 2D-aware Cholesky mesh.
- Steps per system:
  1. Reference THC (generate_isdf_gdf.py): ov c_ref ~12-20, full (bare) c_ref ~24-36.
  2. Exchange: choose c via c^1.5 * norm_loss on a small and a large nk; scan alpha so the
     target (3e-2) is bracketed at every nk; interpolate (log-log).
  3. Optional spot check: custom_mp2_thc on one 2D system that 3e-2 gives ~0.005 eV per
     active electron.
  4. Direct (bare ERI in 2D): choose c so ovvo-projected error <= target; check on 2-3 nk
     that it is nk-independent (not yet verified in 2D, where q=0 has the large negative
     channel).
  5. Check the q-resolved error; if q=0 is problematic, fit q!=0 by THC and treat q=0
     separately.
  6. Report ovvo / oooo / vvvv norms per nk; extrapolate log norm vs log nk.
- Redo MoSe2bulk ISDF if it is still needed.
