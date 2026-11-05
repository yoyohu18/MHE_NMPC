# NTU MAE MSc Dissertation — MHE–NMPC Working Draft

This is the [doem97/NTU-EEE-MSc-Dissertation-Template](https://github.com/doem97/NTU-EEE-MSc-Dissertation-Template)
(originally built for the EEE school), adapted for **NTU School of Mechanical and
Aerospace Engineering (MAE)** MSc dissertations.

## Current dissertation structure

The template has been restructured around the payload-adaptive quadrotor project:

- Chapter 1: problem, objectives, scope and contributions;
- Chapter 2: NMPC, MHE, adaptive control and observability background;
- Chapter 3: 13-state rigid-body model, prior-free total-mass MHE and parcel-mass prediction;
- Chapter 4: cold-start, signed pickup/drop detection, drop regression and adaptive NMPC;
- Chapter 5: ROS 2/PX4/Gazebo implementation, prior-free verification and inherited SITL evidence;
- Chapter 6: algorithmic findings, evidence boundary, limitations and future work;
- appendices: solver configuration, run-record template and submission checklist.

## Template changes

- **Cover page & title page** (`Title/coverpage.tex`, `Title/titlepage.tex`): school name
  changed from "SCHOOL OF ELECTRICAL AND ELECTRONIC ENGINEERING" to
  "SCHOOL OF MECHANICAL AND AEROSPACE ENGINEERING".
- **Year placeholder**: updated from `2021` to `2026` — still a placeholder, change it to
  your actual submission year.
- **Font/margins verified against MAE's current guideline** (no change needed):
  - Font: Times New Roman, 12pt (`mathptmx` package) — matches.
  - Margins: MAE requires "at least 3cm" on all sides. This template already uses
    top=3cm, bottom=3cm, left=3.5cm, right=3cm — already compliant.
  - Line spacing: template uses 1.7, within MAE's "1.5 to double spacing" requirement.

## What you still need to check/fill in yourself

1. **Fill in central metadata** in `main.tex`: `\thesisauthor`,
   `\degreeprogramme` and, if necessary, `\submissionyear`. The dissertation title is
   also defined there once and reused by both title pages and the PDF metadata.
2. **Replace draft-only content**: supervisor/laboratory placeholders and red evidence
   audit notes. The adopted algorithm is aligned to private MHE–NMPC feature revision
   `f2264b3` and explicitly enables no-mass-prior cold start, residual step detection,
   online inertia tracking and drop regression. The thesis distinguishes the prior-free
   MHE from the controller's mass-valued safety floor/envelope. Run and freeze the new experiment matrix before
   setting `\showdraftnotesfalse` in `main.tex`; the older `e003d339` numbers are
   retained only as predecessor evidence.
3. **Verify declaration statement wording** (`Title/SoO.tex`, `Title/SDS.tex`,
   `Title/AAS.tex`): these currently carry the original template's wording for the
   Statement of Originality, Supervisor Declaration Statement, and Authorship
   Attribution Statement. I could not extract the text of MAE's own
   "Thesis Declaration Statements" Word document (binary format, not readable by the
   fetch tool) to confirm it's word-for-word identical — compare against MAE's own copy
   before submission:
   https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/thesis-declaration-statements_25nov2025.docx
4. **Check the current guidelines PDF** for anything that may have changed since this
   was built (Aug 2026):
   https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/guidelines-for-students-writing-msc-dissertations(v-2_250226).pdf
5. **MAE dissertation checklist / submission form**, also worth a look before you submit:
   - https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/ntu-mae-dissertation-check-list(v-1_11122025)4282b2dc-286d-4314-8194-fb9b4f55b7c2.pdf
   - https://www.ntu.edu.sg/media/docs/librariesprovider122/curriculum/dissertation-submission-form(v-1_14112025)523033e7-e42e-455f-87e1-b2e5d4d455e1.pdf

## How to compile

- **Overleaf**: upload the whole folder as a new project, set `main.tex` as the main
  document, and compile with pdfLaTeX.
- **Locally**: use pdfLaTeX and BibTeX.
  ```
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

The working draft should be recompiled after every evidence update. Draft notes remain
visible by design until the prior-free cold-start, payload-prediction and drop-regression
logs have been frozen and audited.

## Not officially endorsed

This is still an unofficial, community-derived template — MAE's Graduate Studies
Office does not publish a LaTeX template of its own (only Word). Treat this as a
formatting shortcut, and do a final visual check against MAE's official guidelines
above before you submit.
