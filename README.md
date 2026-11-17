# MS-Lesion-Burden

Measure multiple sclerosis lesion burden from a T1-weighted MRI and test how it relates
to functional connectivity (FC), urodynamic findings and bladder symptoms. Lesions are
segmented with MindGlide, lesion volume is adjusted for head size, and lesion load is
measured inside the anterior thalamic radiation (ATR) and the superior longitudinal
fasciculus (SLF) of the JHU white-matter atlas. One config file, one command.

A run writes a per-subject table that stays on your machine and a group summary with
no subject rows that can be shared. A viewer then shows each T1 with its lesion overlay
in three linked slice views, every scan's measurements, and the results for each question.

Research use only. Nothing here is a medical device or meant for clinical decisions.

## Install

Python 3.11 to 3.13, and Node.js 20 or newer for the viewer.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
git config core.hooksPath scripts/hooks
```

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
git config core.hooksPath scripts/hooks
```

The `git config` line turns on the commit check that keeps data out of the repository.

With an NVIDIA GPU, install the CUDA build of PyTorch **before** `pip install -e .`,
otherwise Windows gets the CPU-only build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Without a GPU everything still runs; segmentation takes a few minutes per scan instead of
seconds. The first run downloads the MindGlide weights (about 123 MB), the MNI template
and the JHU atlas into local caches.

## Run

1. Copy `config.example.yaml` to `config.yaml`. The default paths work as they are.
2. Put the T1 scans in `data/t1/`, as `<id>.nii.gz` or `<id>/T1w.nii.gz`. Convert DICOM
   to NIfTI first.
3. Fill `data/clinical.csv` and `data/fc.csv` from the templates in `templates/`.
   `data_dictionary.csv` explains every column. The subject IDs must match the scan names.
4. Run it, then look at it:

```bash
mslb run -c config.yaml
mslb view -c config.yaml
```

`mslb view` opens the viewer in your browser, served from your own machine only. The
first time, it builds the viewer, which takes a minute.

- **Scans** shows a T1 with the lesion mask, the two tracts and a crosshair, with the
  scan's measurements beside it: lesion volume, volume adjusted for head size, number
  of lesions, and lesion volume inside the ATR and the SLF.
- **Group results** lists those measurements for every scan and the result for each
  question. Both can be copied as text from the buttons at the top.

The same results are on disk:

| Where | What | May it leave your machine? |
|---|---|---|
| `outputs/local/subjects.csv` | one row per subject: lesion volume and count, head-size factor, tract lesion load, plus your clinical and FC columns | No |
| `outputs/local/subjects/<id>/` | label image, lesion mask, registration, tract labels | No |
| `outputs/shareable/group_summary.md` | counts, means and SDs, and the test results for each question | Yes |
| `outputs/shareable/group_summary.csv`, `descriptives.csv`, `group_summary.json` | the same, in full, for further use | Yes |

[How to read the results](docs/interpreting-results.md) explains every number. A second
run reuses finished segmentations and registrations; `--force` redoes them.

Other commands:

| Command | What it does |
|---|---|
| `mslb run -c config.yaml --dry-run` | check the config and input paths, then stop |
| `mslb run -c config.yaml --view` | run, then open the viewer |
| `mslb show-config -c config.yaml` | print the config with every default filled in |
| `mslb make-fake-data -c config.yaml` | write fake clinical and FC tables for the scans in the folder, to try the pipeline. A summary built on them is stamped FAKE DATA |
| `mslb validate-openms -c config.yaml` | download the public Open MS scans and compare MindGlide on T1 alone with the expert masks |
| `mslb fetch -c config.yaml` | download the template and the atlas ahead of time |
| `mslb check-data` | check that no data file is tracked by git |

To try everything without data of your own:

```bash
mslb validate-openms -c config.yaml
```

Then set `paths.t1_dir` to `data/openms` in `config.yaml` and run
`mslb make-fake-data -c config.yaml`, `mslb run -c config.yaml` and `mslb view -c config.yaml`.

### Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
npm --prefix viewer ci
npm --prefix viewer test
```

### Keeping data out of git

Scans, tables and everything derived from them live in `data/`, `outputs/` and `cache/`,
all git-ignored, and the viewer reads its files from `outputs/local/`. The commit check
also refuses imaging files, spreadsheets, CSV files outside `templates/`, the local
`config.yaml` and files over 1 MB, and refuses direct commits on `main`.

### The public demo site

The viewer on GitHub Pages shows a few scans from the public Open MS dataset. It lives
on the `gh-pages` branch, which holds only the built site. `main` never contains the
demo scans, so cloning `main` to work on patient data does not bring them along
(`git clone --single-branch` skips the `gh-pages` branch entirely).

To rebuild it, export only the scans named in `viewer/public-subjects.json` into
`viewer/public/data/`, which git ignores, and deploy:

```bash
mslb export-viewer -c config.yaml --out viewer/public/data --subjects patient01 patient07 patient12 patient19 --attribution "Scan from the Open MS dataset (Lesjak et al. 2017), CC-BY. Lesion masks by MindGlide."
npm --prefix viewer run deploy
```

The deploy refuses to run if that folder holds a scan that is not on the list, or group
results that were not computed on fake tables. Patient scans never belong on the list.

## Licences and attributions

The code in this repository is under the [MIT licence](LICENSE).

- **MindGlide** (MIT) does the lesion segmentation. Goebl P, Wingrove J, Abdelmannan O,
  et al. Enabling new insights from old scans by repurposing clinical MRI archives for
  multiple sclerosis research. *Nature Communications* 2025;16(1):3149.
  doi:10.1038/s41467-025-58274-8. https://github.com/MS-PINPOINT/mindGlide
- **Open MS dataset** (CC-BY) is used for validation and for the public demo site.
  Lesjak Ž, et al. A novel public MR image dataset of multiple
  sclerosis patients with lesion segmentations based on multi-rater consensus.
  *Neuroinformatics* 2018;16:51-63 (published online 2017). doi:10.1007/s12021-017-9348-7.
  The files are downloaded from https://github.com/muschellij2/open_ms_data.
- **JHU white-matter tractography atlas** supplies the ATR and SLF. Hua K, et al.
  *NeuroImage* 2008;39:336-347. Wakana S, et al. *NeuroImage* 2007;36:630-644. Mori S,
  et al. *MRI Atlas of Human White Matter*, Elsevier, 2005. The atlas is distributed with
  FSL under the [FSL licence](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html), which
  allows **non-commercial use only**. It is not included in this repository: the
  pipeline downloads it into your local cache on first use.
- **MNI152 template** (nonlinear 6th generation, `MNI152NLin6Asym`) is fetched through
  [TemplateFlow](https://www.templateflow.org).
- **ANTs**, through ANTsPy (Apache 2.0), does the registration.
- The head-size adjustment follows the atlas scaling factor of Buckner RL, et al.
  *NeuroImage* 2004;23:724-738.
- The viewer uses three.js (MIT) and IBM Plex Sans (SIL Open Font License).
