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

Produces `main.pdf` (124 pp.). Requires `amsmath`, `amssymb`, `bm`, `siunitx`,
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
| `Figures/` | Imported from `paper/figs/`; regenerate there, not here |

`Chapter*/Chapter*.tex`, `Front/Abstract.tex`, `Front/Acronyms.tex`,
`Front/Symbols.tex` and `Appendix/appendix.tex` are the **superseded** drafts
from before the 2026-09-18 rewrite. They are no longer included by `main.tex`
and are retained only for reference.

## Correspondence with the implementation

The text describes the **deployed defaults** of `offboard_test_acados`, not the
switchable ablations. The load-bearing ones:

| Thesis | Code |
|---|---|
| 16-state MHE, $x_{\rm aug}=[x;m_T;s_x;s_y]$ | `mhe_params.estimate_moment=1`, `ns=2` |
| Frozen inertia coefficient $A$, Eq. (3.34) | `mhe_params.moment_a_mode='frozen'` |
| Window force-balance seed, Eq. (3.47) | `seed_from_thrust=1`, `seed_window_balance=1` |
| Mass box $[0.95\,m_B,\ 5.0]$ | `no_mass_prior=0` (prior-free box is the **ablation**) |
| Stage weights $\beta_j\equiv1$ | `forgetting_lambda=1.0`; event trigger retired |
| Release rule, Eq. (4.24)–(4.25) | `payload_estimate.release_decision` |
| No-payload evidence score $\gamma$, label `eq:gamma` | `payload_estimate.no_payload_evidence_score` (`mode="mass"`) |
| Payload frame, Eq. (4.14) | `payload_estimate.PayloadEstimate` |

Every experimental number in Chapter 5 is taken from the frozen campaign already
reported in `paper/main.tex` and `paper/REPRODUCE.md`; no experiment was re-run
for this document.

## Before submission

See Appendix C. In short: fill the candidate/programme placeholders in
`main.tex` and `Title/`, and set `\showdraftnotesfalse`.
