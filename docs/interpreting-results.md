# How to read the results

Everything below refers to the files in `outputs/shareable/` unless it says otherwise.

## What is measured

| Column | Meaning |
|---|---|
| `lesion_volume_ml` | Volume of everything MindGlide labels as lesion on the T1, after dropping specks smaller than `min_lesion_mm3`. |
| `lesion_count` | Number of separate lesions in that same mask. Voxels that touch, even by a corner, are one lesion. |
| `head_scale` | Head-size factor: template volume divided by the subject's head volume, from the affine part of the registration to the MNI template. Above 1 means a head smaller than the template. Values around 1.1 to 1.6 are normal, because the template is larger than most heads. |
| `lesion_volume_norm_ml` | `lesion_volume_ml × head_scale`. The lesion volume the subject would have if their head were the size of the template. This is what "adjusted for head size" means, and it is the default in the tests. |
| `etiv_ml` | Template brain-mask volume divided by `head_scale`. A rough stand-in for intracranial volume, for orientation only. The tests use `head_scale`, not this. |
| `tract_ATR_ml`, `tract_SLF_ml` | Lesion volume lying inside the tract, both sides added. `_norm_ml` is the head-size-adjusted version. `_pct` is the share of the tract that is lesioned. |
| `tract_*_L`, `_R`, `SLF_temporal` | The same per side, and for the temporal part of the SLF. Reported, not tested. |
| `reg_similarity`, `reg_ok` | A check on the registration. When `reg_ok` is false the head-size and tract values for that subject are left empty and the subject drops out of the tests that need them. |

Set `stats.volume_scale: raw` in the config to test raw volumes instead of adjusted ones.

## The tests

Every test is rank-based and two-sided. Subjects with a missing value are left out of
that one test, and the `n` column says how many went in.

**Spearman correlation** (lesion burden against FC, and against symptom scores).
`estimate` is rho, from -1 to +1. Positive means the two rise together. The interval is
a 95% bootstrap interval: if it spans zero, the data are compatible with no association.
The p-value uses the usual t approximation, which is rough below about ten subjects.

**Mann-Whitney test** (lesion burden in two groups). `estimate` is the rank-biserial
correlation: take every pair of one subject from each group, and it is the share of pairs
where group 1 has the larger value minus the share where group 0 does. +1 means every
group 1 subject is above every group 0 subject, 0 means no separation. `shift` is the
Hodges-Lehmann estimate, the typical difference between the groups in mL (group 1 minus
group 0), with a 95% bootstrap interval. Group 1 is "DO present" or "responder".

**p and q.** `p` is the unadjusted p-value. For the FC questions many connections are
tested at once, so `q` is the Benjamini-Hochberg false discovery rate, computed across
connections within one exposure and one bladder state (for example: total lesion volume,
full bladder). Read FC results by `q`: among results with q below 0.05, about 5% are
expected to be false alarms. The other questions report `p` only, with no adjustment
across outcomes or exposures, so a single p just under 0.05 among a dozen tests is weak
evidence.

**Status.** A test that could not be run says why (`skipped: n below 6`, `skipped: no
variation`) and has no numbers. It is listed so that nothing disappears silently.

**n above zero.** Tract lesion load is often zero. When most subjects have none, a
correlation rests on the few who do, and should be read with that in mind.

## The questions

| | Exposure or grouping | Outcome | Test |
|---|---|---|---|
| Q1 | total lesion volume (primary), lesion count (secondary) | each FC connection, in the full state, the empty state, and the change | Spearman, FDR across connections |
| Q2 | ATR and SLF lesion volume | the same FC outcomes | Spearman, FDR across connections |
| Q3 | detrusor overactivity present or absent | total, ATR and SLF lesion volume | Mann-Whitney |
| Q4 | total, ATR and SLF lesion volume | NBSS total (primary), three bladder-diary measures (secondary) | Spearman |
| Q5 | responder or non-responder, active arm only | baseline total, ATR and SLF lesion volume | Mann-Whitney |

FC change is the full-bladder value minus the empty-bladder value. The FC questions are
about subjects scanned in both bladder states, so a subject enters an FC test for a
connection only when both their full and their empty value are there (or a change value
was supplied directly). `stats.fc_require_both_states: false` lifts that rule. Q5 is off
unless the config turns it on and sets the response threshold.

## Where to find things

`mslb view` shows all of this in the browser. The **Scans** side has one scan's
measurements beside its images. **Group results** has the table of every scan's
measurements, the cohort summary, and one section per question, in the same order as the
table above. Both can be copied as text.

## What the summary can and cannot say

- These are associations in one sample. They do not show that lesions cause anything.
- With a small sample the intervals are wide. An interval that spans zero is not evidence
  of no effect, only of not enough data to tell.
- The lesion mask comes from T1 alone. T1 shows fewer lesions than FLAIR, and shows them
  smaller. On the public Open MS scans, MindGlide's T1-only volume ranks subjects much as
  expert masks do, while voxel-by-voxel overlap with those masks is modest
  (`mslb validate-openms` reproduces the numbers). Volumes are therefore better trusted
  for ordering subjects than as absolute amounts, which is what the rank-based tests use.
- The tracts are atlas regions warped onto each subject, not that subject's own tracts.

## What may be shared

`outputs/shareable/` holds counts, means, SDs, medians, quartiles and test results. It has
no subject IDs, no per-subject values and no minimum or maximum (those are one person's
value). Statistics for a group smaller than `report.min_cell_n` are withheld. Everything
in `outputs/local/` is per subject and stays where it is.
