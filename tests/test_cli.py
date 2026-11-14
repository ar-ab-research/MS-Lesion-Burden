import pytest
import yaml

from mslb import __version__
from mslb.cli import main


def make_inputs(tmp_path):
    (tmp_path / "data" / "t1").mkdir(parents=True)
    (tmp_path / "data" / "clinical.csv").write_text("subject_id\n")
    (tmp_path / "data" / "fc.csv").write_text("subject_id\n")
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    return config


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_dry_run_passes_with_inputs_in_place(tmp_path, capsys):
    config = make_inputs(tmp_path)
    assert main(["run", "-c", str(config), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "segment" in out and "q1_total_burden_fc" in out
    assert "q5_treatment_response" not in out


def test_run_names_missing_inputs(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    assert main(["run", "-c", str(config), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "t1_dir" in err and "clinical_csv" in err and "fc_csv" in err


def test_bad_config_is_reported_not_raised(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("stats: {alhpa: 1}\n")
    assert main(["run", "-c", str(config)]) == 2
    assert "alhpa" in capsys.readouterr().err
    assert main(["run", "-c", str(tmp_path / "missing.yaml")]) == 2


def test_show_config_prints_loadable_yaml_with_defaults(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("stats: {alpha: 0.01}\n")
    assert main(["show-config", "-c", str(config)]) == 0
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["stats"]["alpha"] == 0.01
    assert shown["analyses"]["q5_treatment_response"] is False
    assert shown["paths"]["t1_dir"].endswith("data/t1")
