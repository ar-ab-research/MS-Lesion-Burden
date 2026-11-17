import json
import os
import threading
import urllib.error
import urllib.request

import pytest

from mslb import view, viewer_export
from mslb.cli import main
from mslb.config import from_dict


@pytest.fixture
def served(tmp_path):
    """A stand-in built viewer and a data folder, served the way mslb view serves them."""
    site = tmp_path / "dist"
    data = tmp_path / "viewer_data"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<html>viewer</html>")
    (site / "assets" / "app.js").write_text("console.log(1)")
    # demo data baked into a build must never be what a local run shows
    (site / "data").mkdir()
    (site / "data" / "index.json").write_text('{"subjects": ["demo"]}')
    (data / "s1").mkdir(parents=True)
    (data / "index.json").write_text('{"subjects": ["s1"]}')
    (data / "s1" / "meta.json").write_text('{"id": "s1"}')
    (tmp_path / "secret.txt").write_text("not served")
    server = view.make_server(site, data, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode(), response.headers


def test_the_app_and_the_scans_come_from_their_own_folders(served):
    assert "viewer" in get(served + "/")[0]
    assert "console.log" in get(served + "/assets/app.js")[0]
    assert json.loads(get(served + "/data/index.json")[0]) == {"subjects": ["s1"]}
    assert json.loads(get(served + "/data/s1/meta.json?x=1")[0]) == {"id": "s1"}


def test_nothing_is_cached_and_nothing_outside_the_two_folders_is_served(served):
    assert get(served + "/data/index.json")[1]["Cache-Control"] == "no-store"
    for path in ("/data/../secret.txt", "/../secret.txt", "/data/missing.json"):
        with pytest.raises(urllib.error.HTTPError) as info:
            get(served + path)
        assert info.value.code == 404


def test_only_this_machine_can_reach_it(tmp_path):
    server = view.make_server(tmp_path, tmp_path, port=0)
    assert server.server_address[0] == "127.0.0.1"
    server.server_close()


def test_a_taken_port_moves_to_the_next_one(tmp_path):
    first = view.make_server(tmp_path, tmp_path, port=0)
    port = first.server_address[1]
    second = view.make_server(tmp_path, tmp_path, port=port)
    assert second.server_address[1] != port
    first.server_close()
    second.server_close()


def test_build_is_needed_when_missing_or_older_than_the_source(tmp_path):
    viewer = tmp_path / "viewer"
    (viewer / "src").mkdir(parents=True)
    for name in ("index.html", "package.json", "src/main.js"):
        (viewer / name).write_text("x")
    assert view.needs_build(viewer)
    (viewer / "dist").mkdir()
    built = viewer / "dist" / "index.html"
    built.write_text("built")
    later = built.stat().st_mtime + 10
    os.utime(built, (later, later))
    assert not view.needs_build(viewer)
    os.utime(viewer / "src" / "main.js", (later + 10, later + 10))
    assert view.needs_build(viewer)


def test_without_node_the_message_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(view.shutil, "which", lambda name: None)
    with pytest.raises(view.ViewerError, match="Node.js.*group_summary.md"):
        view.build(tmp_path)


def test_view_before_any_run_explains_instead_of_serving_nothing(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    (tmp_path / "data" / "t1").mkdir(parents=True)
    monkeypatch.setattr(view, "build", lambda *a, **k: pytest.fail("should not build"))
    assert main(["view", "-c", str(config), "--no-browser"]) == 2
    assert "run the pipeline first" in capsys.readouterr().err


def test_default_export_folder_is_with_the_local_outputs(tmp_path):
    cfg = from_dict({}, base=tmp_path)
    out = viewer_export.default_out(cfg)
    assert out == cfg.paths.local_dir / "viewer"
    assert "public" not in out.parts
