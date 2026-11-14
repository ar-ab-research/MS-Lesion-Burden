from pathlib import Path

import pytest
import yaml

from mslb.config import Config, ConfigError, check_inputs, from_dict, load

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config.example.yaml"


def write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_example_file_spells_out_exactly_the_defaults():
    # the example is the documentation of the defaults, so the two must not drift
    example = load(EXAMPLE)
    defaults = from_dict({}, base=ROOT)
    assert example.as_dict() == defaults.as_dict()
    # and it names every setting, so nothing is configurable but undocumented
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    for section, values in defaults.as_dict().items():
        assert set(raw[section]) == set(values), section


def test_defaults_match_the_agreed_design():
    cfg = from_dict({})
    assert cfg.analyses.q5_treatment_response is False
    assert cfg.enabled_analyses() == [
        "q1_total_burden_fc", "q2_tract_load_fc", "q3_detrusor_overactivity", "q4_symptom_severity",
    ]
    assert cfg.icv.method == "affine_scaling"
    assert cfg.segmentation.lesion_label == 18
    assert set(cfg.tracts.primary) == {"ATR", "SLF"}
    assert not any("temporal" in label for labels in cfg.tracts.primary.values() for label in labels)
    assert cfg.treatment_response.min_nbss_improvement is None


def test_relative_paths_resolve_against_the_config_file(tmp_path, monkeypatch):
    path = write(tmp_path, {"paths": {"t1_dir": "scans", "output_dir": "out"}})
    monkeypatch.chdir(ROOT)  # not the config's folder
    cfg = load(path)
    assert cfg.paths.t1_dir == (tmp_path / "scans").resolve()
    assert cfg.paths.local_dir == (tmp_path / "out" / "local").resolve()
    assert cfg.paths.shareable_dir == (tmp_path / "out" / "shareable").resolve()
    assert cfg.source == path.resolve()


def test_absolute_paths_are_kept(tmp_path):
    scans = (tmp_path / "elsewhere").resolve()
    cfg = load(write(tmp_path, {"paths": {"t1_dir": str(scans)}}))
    assert cfg.paths.t1_dir == scans


def test_a_partial_config_keeps_the_other_defaults(tmp_path):
    cfg = load(write(tmp_path, {"stats": {"alpha": 0.01}}))
    assert cfg.stats.alpha == 0.01
    assert cfg.stats.bootstrap_resamples == Config().stats.bootstrap_resamples


@pytest.mark.parametrize("data, needle", [
    ({"pathz": {}}, "pathz"),                                         # misspelt section
    ({"stats": {"alhpa": 0.01}}, "alhpa"),                            # misspelt setting
    ({"stats": {"alpha": 1.5}}, "stats.alpha"),
    ({"stats": {"alpha": "0.05"}}, "stats.alpha"),                    # text where a number goes
    ({"stats": {"seed": 1.5}}, "stats.seed"),
    ({"stats": {"min_n": True}}, "stats.min_n"),                      # yaml `yes` is not a count
    ({"segmentation": {"device": "gpu"}}, "segmentation.device"),
    ({"registration": {"mask_lesions": "yes please"}}, "registration.mask_lesions"),
    ({"icv": {"method": "external_csv"}}, "icv.external_csv"),
    ({"analyses": {"q5_treatment_response": True}}, "min_nbss_improvement"),
    ({"treatment_response": {"min_nbss_improvement": -1}}, "min_nbss_improvement"),
    ({"tracts": {"primary": {}}}, "tracts.primary"),
    ({"tracts": {"primary": {"ATR": "Anterior thalamic radiation L"}}}, "tracts.primary"),
    ({"tracts": {"secondary": {"ATR": ["Anterior thalamic radiation L"]}}}, "both primary and secondary"),
    ({"paths": {"t1_dir": ""}}, "paths.t1_dir"),
    ({"paths": None, "stats": []}, "stats"),
    (["not", "a", "mapping"], "mapping"),
])
def test_bad_configs_are_rejected_with_the_setting_named(tmp_path, data, needle):
    with pytest.raises(ConfigError, match=needle):
        load(write(tmp_path, data))


def test_q5_turns_on_once_the_threshold_is_given(tmp_path):
    cfg = load(write(tmp_path, {
        "analyses": {"q5_treatment_response": True},
        "treatment_response": {"min_nbss_improvement": 5},
    }))
    assert "q5_treatment_response" in cfg.enabled_analyses()
    assert cfg.treatment_response.min_nbss_improvement == 5.0


def test_missing_file_and_broken_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("paths: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid yaml"):
        load(bad)


def test_empty_file_is_all_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load(path).as_dict() == from_dict({}, base=tmp_path).as_dict()


def test_digest_tracks_settings_not_formatting(tmp_path):
    a = load(write(tmp_path, {"stats": {"alpha": 0.05}}))
    b = load(write(tmp_path, {}))
    c = load(write(tmp_path, {"stats": {"alpha": 0.01}}))
    assert a.digest() == b.digest() != c.digest()


def test_check_inputs_lists_what_is_missing_then_clears(tmp_path):
    cfg = load(write(tmp_path, {}))
    problems = check_inputs(cfg)
    assert len(problems) == 3
    assert any("t1_dir" in p for p in problems)
    (tmp_path / "data" / "t1").mkdir(parents=True)
    (tmp_path / "data" / "clinical.csv").write_text("subject_id\n")
    (tmp_path / "data" / "fc.csv").write_text("subject_id\n")
    assert check_inputs(cfg) == []
