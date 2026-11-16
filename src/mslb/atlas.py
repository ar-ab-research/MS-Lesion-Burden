"""The template and the tract atlas. Both are downloaded on first use, never shipped.

The JHU white-matter tractography atlas is distributed with FSL under FSL's
licence, which allows non-commercial use only. It is fetched into the local cache
and is not part of this repository. Tracts are found by the names in the atlas's
own XML file, not by fixed numbers.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from mslb.config import Config
from mslb.download import fetch
from mslb.errors import PipelineError

JHU_BASE = "https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/raw/master"
JHU_FILES = {
    "labels": ("JHU/JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz", "JHU-ICBM-tracts-maxprob-thr25-1mm.nii.gz"),
    "xml": ("JHU-tracts.xml", "JHU-tracts.xml"),
}
JHU_NOTICE = (
    "Downloading the JHU white-matter tractography atlas (Hua et al. 2008; Wakana et al. 2007;\n"
    "Mori et al. 2005) from the FSL atlas repository. It is released under the FSL licence:\n"
    "non-commercial use only. It stays in the local cache and is not redistributed here."
)


class AtlasError(PipelineError):
    """The atlas is missing a tract, or could not be read."""


def jhu_paths(cfg: Config, log=print) -> dict[str, Path]:
    """Local paths of the atlas label image and its XML, downloading them if needed."""
    folder = cfg.paths.cache_dir / "jhu"
    paths = {key: folder / name for key, (_, name) in JHU_FILES.items()}
    if not all(p.is_file() for p in paths.values()):
        log(JHU_NOTICE)
        for key, (remote, _) in JHU_FILES.items():
            fetch(f"{JHU_BASE}/{remote}", paths[key])
    return paths


def read_tract_names(xml_path: Path) -> dict[str, int]:
    """Tract name to its value in the max-probability label image.

    The XML numbers tracts from 0. The label image keeps 0 for background, so a
    tract's voxels carry its XML index plus one.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise AtlasError(f"{xml_path}: not a readable atlas description ({exc})") from exc
    names = {}
    for label in root.iter("label"):
        names[(label.text or "").strip()] = int(label.attrib["index"]) + 1
    if not names:
        raise AtlasError(f"{xml_path}: no tract labels found")
    return names


def tract_values(groups: dict[str, list[str]], names: dict[str, int]) -> dict[str, list[int]]:
    """Label values for each named group of tracts. Unknown names are an error."""
    out = {}
    for group, wanted in groups.items():
        missing = [w for w in wanted if w not in names]
        if missing:
            raise AtlasError(
                f"tract(s) not in the atlas: {', '.join(missing)}. "
                f"available: {', '.join(sorted(names))}"
            )
        out[group] = [names[w] for w in wanted]
    return out


def template_paths(cfg: Config) -> dict[str, Path]:
    """Whole-head T1 template and its brain mask, through TemplateFlow."""
    os.environ.setdefault("TEMPLATEFLOW_HOME", str(cfg.paths.cache_dir / "templateflow"))
    from templateflow import api

    name = cfg.registration.template
    t1 = api.get(name, resolution=1, suffix="T1w", desc=None, extension=".nii.gz")
    mask = api.get(name, resolution=1, suffix="mask", desc="brain", extension=".nii.gz")
    for what, got in (("T1", t1), ("brain mask", mask)):
        if isinstance(got, list) or not Path(got).is_file():
            raise AtlasError(f"could not get one {what} image for template {name}")
    return {"t1": Path(t1), "mask": Path(mask)}


def fetch_all(cfg: Config, log=print) -> dict[str, Path]:
    paths = dict(jhu_paths(cfg, log=log))
    paths.update({f"template_{k}": v for k, v in template_paths(cfg).items()})
    return paths
