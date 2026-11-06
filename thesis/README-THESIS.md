# MSc dissertation source (NTU MAE template)

Single-column thesis version of this project. The two-column ICRA manuscript
lives one directory up in [`paper/`](../); the two diverge on purpose — the
conference version is condensed, this one carries the full derivations.

## Build

```bash
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Produces `main.pdf`. Requires `amsmath`, `amssymb`, `bm`, `siunitx`,
`booktabs`, `tabularx`, `tikz`, `mathptmx`. Notation macros (`\mB`, `\mT`,
`\svec`, `\dJ`, `\Tphys`, …) are defined once in the `main.tex` preamble and are
shared by every chapter.

## Layout

| File | Contents |
|---|---|
| `Chapter1/ch_intro.tex` | Problem statement, objectives, scope, contributions |
| `Chapter2/ch_related.tex` | Predictive control, MHE, inertial-parameter ID, change detection |
| `Chapter3/ch_model.tex` | Model and parameter structure — the derivations |
| `Chapter4/ch_framework.tex` | Both OCPs, the payload interface, the eventless lifecycle |
| `Chapter5/ch_eval.tex` | Implementation and the frozen SITL campaign |
| `Chapter6/ch_conclusion.tex` | Findings, limitations, future work |
| `Front/*_new.tex` | Abstract, acronyms, symbols |
| `Appendix/appendix_new.tex` | OCP configuration tables, record template, checklist |
| `Figures/` | Imported from `paper/figs/` (regenerate there), except the deployed-system figures, whose sources are here: `fig_lifecycle_fsm_deployed.tex` (`pdflatex`) and `fig_architecture_deployed.py` (`python3`, writes `fig1_architecture_deployed.*`) |

`Chapter*/Chapter*.tex`, `Front/Abstract.tex`, `Front/Acronyms.tex`,
`Front/Symbols.tex` and `Appendix/appendix.tex` are the **superseded** drafts
from before the 2026-09-18 rewrite. They are no longer included by `main.tex`
and are retained only for reference.

## Correspondence with the implementation

The text describes the **deployed system** of the mainline code base
(`../src/offboard_test_acados/offboard_test_acados/`, commit 5b7881e and later),
whose parameter defaults are the deployed values. Every reported flight was run
from the campaign code base (`ros2_ws_HJH`), which holds the switchable
alternatives (full-state estimator F, release arms A/B/C, fixed and L1
controllers); Chapter 5 states which configuration each result belongs to.

| Thesis | Code (mainline) |
|---|---|
| Reduced estimator, $x_R=[v;m_T]$, Eq. `eq:xred`/`eq:redmodel` | `mhe/model.py`, `mhe/params.py` |
| Window force-balance seed, `eq:winbal` | `mhe/node.py` `_window_balance_mass` |
| Mass box $[0.70\,m_B, 5.0]$ and health band `eq:healthband` | `mhe/params.py`, `mhe/estimator.py` `MassHealthMonitor` |
| Residual-triggered de-weighting (Table `tab:stageweight`) | `mhe/weights.py`, `mhe/node.py` `_residual_detect` |
| Payload frame `eq:frame` and mass score $\gamma$ `eq:gamma` | `payload_estimate.py` (`PayloadEstimate`, `no_payload_evidence_score`, mode `mass`) |
| NMPC model `eq:predmodel`, $p\in\mathbb R^{20}$ `eq:pnmpc` | `acados_model.py`, `acados_params.py` |
| Controller-side filter/slew `eq:lpfslew`, attach-time floor | `acados_nmpc_node.py` `_update_continuous_payload_model` |
| Residual specific force `eq:l1resid` | `l1_adaptive.py`, `acados_nmpc_node.py` `_l1_update` |
| Body-rate scaling `eq:omegascale`–`eq:headroomapply` | `acados_nmpc_node.py` `_update_omega_scale`, `payload_estimate.headroom_limited_scale` |
| Force-balance release evidence `eq:macc`/`eq:gammaacc` (arm D) | `acc_release_detector.py` |
| Confirmation `eq:confirm`, dual enabling, loss watchdog, UNRESOLVED brake | `acados_nmpc_node.py` `_confirm_no_payload_if_persistent`, `_update_acc_loss_watchdog`, `_begin_unresolved_hover` |

Every experimental number in Chapter 5 is taken from the frozen campaign already
reported in `paper/main.tex` and `paper/REPRODUCE.md`; no experiment was re-run
for this document.

## Before submission

See Appendix C. In short: fill the candidate/programme placeholders in
`main.tex` and `Title/`, and set `\showdraftnotesfalse`.
