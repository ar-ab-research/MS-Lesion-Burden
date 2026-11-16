import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from mslb import atlas, icv
from mslb.config import from_dict
from mslb.tracts import tract_loads

XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<atlas version="1.0"><header><name>JHU White-Matter Tractography Atlas</name></header><data>
<label index="0" x="98" y="117" z="75">Anterior thalamic radiation L</label>
<label index="1" x="83" y="117" z="76">Anterior thalamic radiation R</label>
<label index="2" x="96" y="99" z="36">Corticospinal tract L</label>
<label index="14" x="129" y="112" z="102">Superior longitudinal fasciculus L</label>
<label index="15" x="40" y="121" z="99">Superior longitudinal fasciculus R</label>
<label index="18" x="140" y="89" z="61">Superior longitudinal fasciculus (temporal part) L</label>
<label index="19" x="52" y="116" z="103">Superior longitudinal fasciculus (temporal part) R</label>
</data></atlas>"""


@pytest.fixture
def names(tmp_path):
    path = tmp_path / "JHU-tracts.xml"
    path.write_text(XML, encoding="iso-8859-1")
    return atlas.read_tract_names(path)


def test_label_value_is_xml_index_plus_one(names):
    assert names["Anterior thalamic radiation L"] == 1
    assert names["Superior longitudinal fasciculus R"] == 16
    assert names["Superior longitudinal fasciculus (temporal part) R"] == 20


def test_default_config_tracts_resolve_and_the_main_slf_excludes_the_temporal_part(names):
    cfg = from_dict({})
    primary = atlas.tract_values(cfg.tracts.primary, names)
    assert primary == {"ATR": [1, 2], "SLF": [15, 16]}
    secondary = atlas.tract_values(cfg.tracts.secondary, names)
    assert secondary["SLF_temporal"] == [19, 20] and secondary["ATR_L"] == [1]


def test_unknown_tract_is_an_error_that_lists_what_exists(names):
    with pytest.raises(atlas.AtlasError, match="Fornix.*available: Anterior thalamic"):
        atlas.tract_values({"X": ["Fornix"]}, names)


def test_unreadable_atlas_xml(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<atlas><data>")
    with pytest.raises(atlas.AtlasError):
        atlas.read_tract_names(bad)
    empty = tmp_path / "empty.xml"
    empty.write_text("<atlas><data></data></atlas>")
    with pytest.raises(atlas.AtlasError, match="no tract labels"):
        atlas.read_tract_names(empty)


def write_mat(path, matrix, key="AffineTransform_double_3_3"):
    params = np.concatenate([np.asarray(matrix, float).ravel(), [1.0, -2.0, 3.0]])
    savemat(str(path), {key: params.reshape(12, 1), "fixed": np.zeros((3, 1))}, format="4")
    return path


def test_head_scale_is_template_volume_over_subject_volume(tmp_path):
    # the stored matrix maps template points to subject points. a subject head at 0.8 of
    # the template's width has 0.512 of its volume, so it must be scaled up by 1/0.512
    small = write_mat(tmp_path / "small.mat", np.diag([0.8, 0.8, 0.8]))
    assert icv.head_scale(small) == pytest.approx(1 / 0.512)
    big = write_mat(tmp_path / "big.mat", np.diag([1.1, 1.0, 1.0]), key="AffineTransform_float_3_3")
    assert icv.head_scale(big) == pytest.approx(1 / 1.1)


def test_head_scale_ignores_rotation_and_reflection(tmp_path):
    theta = 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert icv.head_scale(write_mat(tmp_path / "r.mat", rot)) == pytest.approx(1.0)
    assert icv.head_scale(write_mat(tmp_path / "f.mat", np.diag([-1.0, 1.0, 1.0]))) == pytest.approx(1.0)


def test_bad_transforms_are_errors(tmp_path):
    with pytest.raises(icv.IcvError, match="singular"):
        icv.head_scale(write_mat(tmp_path / "s.mat", np.zeros((3, 3))))
    junk = tmp_path / "junk.mat"
    junk.write_bytes(b"not a transform")
    with pytest.raises(icv.IcvError):
        icv.head_scale(junk)


def test_plausible_range():
    assert icv.plausible(1.0) and icv.plausible(1.5)
    assert not icv.plausible(0.4) and not icv.plausible(3.0)


def test_external_icv_table(tmp_path):
    path = tmp_path / "icv.csv"
    pd.DataFrame({"subject_id": ["a", "b", "c"], "icv_ml": [1200, 1500, 1800]}).to_csv(path, index=False)
    cfg = from_dict({"icv": {"method": "external_csv", "external_csv": str(path)}})
    got = icv.external_scales(cfg, ["a", "b", "c"]).set_index("subject_id")
    assert got.loc["b", "head_scale"] == pytest.approx(1.0)       # the mean head
    assert got.loc["a", "head_scale"] == pytest.approx(1.25)      # smaller head, scaled up
    assert got.loc["a", "etiv_ml"] == 1200
    missing = icv.external_scales(cfg, ["a", "zz"]).set_index("subject_id")
    assert np.isnan(missing.loc["zz", "head_scale"])
    pd.DataFrame({"subject_id": ["a"], "volume": [1]}).to_csv(path, index=False)
    with pytest.raises(icv.IcvError, match="icv_ml"):
        icv.external_scales(cfg, ["a"])


def test_tract_loads_known_volumes():
    labels = np.zeros((10, 10, 10), np.int16)
    labels[0:5, 0:2, 0] = 1        # 10 voxels
    labels[5:10, 0:2, 0] = 2       # 10 voxels
    labels[0:4, 5, 5] = 15         # 4 voxels
    lesion = np.zeros((10, 10, 10), bool)
    lesion[0:3, 0, 0] = True       # 3 voxels in label 1
    lesion[7, 0:2, 0] = True       # 2 voxels in label 2
    lesion[9, 9, 9] = True         # outside every tract
    out = tract_loads(labels, lesion, {"ATR": [1, 2], "ATR_L": [1], "SLF": [15, 16]}, voxel_mm3=2.0)
    assert out["tract_ATR_ml"] == pytest.approx(5 * 2 / 1000)
    assert out["tract_ATR_L_ml"] == pytest.approx(3 * 2 / 1000)
    assert out["tract_ATR_size_ml"] == pytest.approx(20 * 2 / 1000)
    assert out["tract_ATR_pct"] == pytest.approx(25.0)
    assert out["tract_SLF_ml"] == 0.0 and out["tract_SLF_pct"] == 0.0


def test_tract_missing_from_the_field_of_view_gives_missing_share():
    out = tract_loads(np.zeros((4, 4, 4), np.int16), np.ones((4, 4, 4), bool), {"SLF": [15]}, 1.0)
    assert out["tract_SLF_ml"] == 0.0 and np.isnan(out["tract_SLF_pct"])


@pytest.mark.slow
def test_ants_affine_recovers_a_known_head_size(tmp_path):
    """The convention check, on real ANTs output: shrink a volume by a known factor,
    register it to the original, and the head-size factor must come out as the
    inverse volume ratio, not the ratio itself."""
    ants = pytest.importorskip("ants")
    from scipy.ndimage import gaussian_filter

    n = 64
    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).astype(float)
    centre = (n - 1) / 2

    def head(scale):
        """Scalp, skull, brain and two off-centre blobs, all scaled about the centre."""
        def ellipsoid(radii, offset=(0, 0, 0)):
            d = (grid - centre - np.array(offset) * scale) / (np.array(radii) * scale)
            return (d ** 2).sum(-1) <= 1

        img = np.zeros((n, n, n), np.float32)
        img[ellipsoid((24, 28, 22))] = 1.0
        img[ellipsoid((21, 25, 19))] = 0.2
        img[ellipsoid((19, 23, 17))] = 0.7
        img[ellipsoid((5, 8, 4), (4, -3, 2))] = 0.3
        img[ellipsoid((3, 3, 3), (-8, 9, -5))] = 1.0
        return gaussian_filter(img, 0.7)

    spacing = (3.0, 3.0, 3.0)
    fixed = ants.from_numpy(head(1.0), spacing=spacing)
    moving = ants.from_numpy(head(0.85), spacing=spacing)
    ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine",
                      outprefix=str(tmp_path / "t_"))
    scale = icv.head_scale(tmp_path / "t_0GenericAffine.mat")
    assert scale == pytest.approx(1 / 0.85 ** 3, rel=0.03)      # 1.63, and not its reciprocal 0.61
