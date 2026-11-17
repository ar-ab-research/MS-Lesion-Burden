"""Open the results in the slice viewer: export what it needs, build it once, serve it locally.

The viewer is a small web app in viewer/. It is served from this machine only
(127.0.0.1), and its scan files are read from outputs/local, so nothing leaves.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mslb import inputs, viewer_export
from mslb.config import Config
from mslb.errors import PipelineError

DEFAULT_PORT = 5280


class ViewerError(PipelineError):
    """The viewer could not be built or started."""


def viewer_dir() -> Path:
    """The viewer's source folder, which sits beside src/ in a clone of the repository."""
    folder = Path(__file__).resolve().parents[2] / "viewer"
    if not (folder / "package.json").is_file():
        raise ViewerError(
            "the viewer's source was not found. it ships with the repository, so install "
            "from a clone with: pip install -e ."
        )
    return folder


def needs_build(viewer: Path) -> bool:
    """True when there is no built viewer, or its source has changed since it was built."""
    built = viewer / "dist" / "index.html"
    if not built.is_file():
        return True
    sources = [viewer / "index.html", viewer / "package.json", *(viewer / "src").glob("*")]
    return any(p.stat().st_mtime > built.stat().st_mtime for p in sources if p.is_file())


def build(viewer: Path, log=print) -> None:
    """Install the viewer's packages if needed and build it. Needs Node.js."""
    npm = shutil.which("npm")
    if npm is None:
        raise ViewerError(
            "the viewer needs Node.js 20 or newer (https://nodejs.org) to be built once. "
            "the results themselves are already in outputs/shareable/group_summary.md"
        )
    steps = [[npm, "run", "build"]]
    if not (viewer / "node_modules").is_dir():
        steps.insert(0, [npm, "ci"])
    for step in steps:
        log(f"viewer    npm {' '.join(step[1:])}")
        done = subprocess.run(step, cwd=viewer, capture_output=True, text=True)
        if done.returncode != 0:
            raise ViewerError(f"npm {' '.join(step[1:])} failed:\n{(done.stderr or done.stdout).strip()[-1500:]}")


class _Handler(SimpleHTTPRequestHandler):
    """Everything under /data/ comes from the exported scans, the rest from the built viewer."""

    site: Path
    data: Path

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/data" or clean.startswith("/data/"):
            self.directory = str(self.data)
            return super().translate_path(clean[len("/data"):] or "/")
        self.directory = str(self.site)
        return super().translate_path(path)

    def end_headers(self) -> None:
        # the files change between runs, a stale copy from the browser's cache would mislead
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


class _Server(ThreadingHTTPServer):
    # on windows the usual reuse flag lets two programs listen on one port at once,
    # and the browser would reach whichever answered first
    allow_reuse_address = os.name != "nt"


def make_server(site: Path, data: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """A server on this machine only. If the port is taken, the next few are tried."""
    handler = type("Handler", (_Handler,), {"site": Path(site), "data": Path(data)})
    last = None
    for candidate in ([0] if port == 0 else range(port, port + 20)):
        try:
            return _Server(("127.0.0.1", candidate), partial(handler, directory=str(site)))
        except OSError as exc:
            last = exc
    raise ViewerError(f"no free port from {port} to {port + 19}: {last}")


def prepare(cfg: Config, log=print) -> tuple[Path, Path]:
    """Export what the viewer shows and make sure the viewer is built. Returns (site, data)."""
    scans = inputs.discover_t1(cfg.paths.t1_dir)
    data = viewer_export.default_out(cfg)
    ids = viewer_export.export(cfg, scans, data, log=log)
    if not ids:
        raise ViewerError("there is nothing to show yet. run the pipeline first: mslb run -c config.yaml")
    viewer = viewer_dir()
    if needs_build(viewer):
        build(viewer, log=log)
    return viewer / "dist", data


def view(cfg: Config, port: int = DEFAULT_PORT, open_browser: bool = True, log=print) -> None:
    """Show the processed scans and the group results in the browser until interrupted."""
    site, data = prepare(cfg, log=log)
    server = make_server(site, data, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    log(f"viewer    {url}   (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
