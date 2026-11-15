"""Fetch a file once and keep it. Used for the atlas and the public validation data."""
from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from mslb.errors import PipelineError


class DownloadError(PipelineError):
    """A file could not be fetched."""


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> Path:
    """Download url to dest unless dest is already there. Returns dest.

    Written to a temporary name first, so an interrupted download never leaves a
    truncated file that a later run would take for the real thing.
    """
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mslb"})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            if partial.stat().st_size == 0:
                raise DownloadError(f"{url} came back empty")
            partial.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last = exc
            partial.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise DownloadError(f"could not download {url}: {last}")
