#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


def is_gcs_url(path: str) -> bool:
    """Return True if *path* looks like a GCS URI (gs://)."""
    return path.startswith("gs://")


class GCSPath:
    """A ``pathlib.Path``-compatible wrapper for GCS URIs.

    Mirrors the subset of the ``pathlib.Path`` interface used by
    :class:`~neptune_exporter.storage.parquet_reader.ParquetReader` and
    :class:`~neptune_exporter.loader_manager.LoaderManager` so both classes
    can work transparently with either local paths or GCS paths.

    ``gcsfs`` is imported lazily the first time a filesystem operation is
    performed so that the class can be imported without the optional ``gcs``
    extra being installed.
    """

    def __init__(self, uri: str, _fs=None) -> None:
        self._uri = uri.rstrip("/")
        self._fs = _fs  # lazily initialised via _get_fs()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fs(self):
        if self._fs is None:
            try:
                import gcsfs  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "gcsfs is required for GCS path support. "
                    "Install it with: uv sync --extra gcs"
                ) from exc
            self._fs = gcsfs.GCSFileSystem()
        return self._fs

    def _raw(self) -> str:
        """Return the path **without** the ``gs://`` scheme for gcsfs/PyArrow."""
        if self._uri.startswith("gs://"):
            return self._uri[len("gs://"):]
        return self._uri

    # ------------------------------------------------------------------
    # Path-like properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._uri.rstrip("/").split("/")[-1]

    @property
    def stem(self) -> str:
        n = self.name
        dot = n.rfind(".")
        return n[:dot] if dot > 0 else n

    @property
    def suffix(self) -> str:
        n = self.name
        dot = n.rfind(".")
        return n[dot:] if dot > 0 else ""

    @property
    def parent(self) -> GCSPath:
        parts = self._uri.rstrip("/").rsplit("/", 1)
        return GCSPath(parts[0] if len(parts) > 1 else self._uri, self._fs)

    # Property expected by pyarrow reader helper
    @property
    def gcs_path(self) -> str:
        """Path without ``gs://`` prefix, suitable for PyArrow filesystem calls."""
        return self._raw()

    # ------------------------------------------------------------------
    # Operators / dunder helpers
    # ------------------------------------------------------------------

    def __truediv__(self, other: str) -> GCSPath:
        other = str(other).lstrip("/")
        return GCSPath(f"{self._uri}/{other}", self._fs)

    def __str__(self) -> str:
        return self._uri

    def __repr__(self) -> str:
        return f"GCSPath('{self._uri}')"

    def __fspath__(self) -> str:
        # Allows os.fspath() to work (needed by some libraries)
        return self._uri

    # ------------------------------------------------------------------
    # pathlib.Path-compatible operations
    # ------------------------------------------------------------------

    def absolute(self) -> GCSPath:
        """No-op: GCS paths are already absolute."""
        return self

    def exists(self) -> bool:
        return self._get_fs().exists(self._raw())

    def is_dir(self) -> bool:
        return self._get_fs().isdir(self._raw())

    def iterdir(self) -> Iterator[GCSPath]:
        fs = self._get_fs()
        try:
            for entry in fs.ls(self._raw(), detail=False):
                yield GCSPath(f"gs://{entry}", fs)
        except FileNotFoundError:
            return

    def glob(self, pattern: str) -> Iterator[GCSPath]:
        fs = self._get_fs()
        full_pattern = f"{self._raw()}/{pattern}"
        for entry in fs.glob(full_pattern):
            yield GCSPath(f"gs://{entry}", fs)

    # ------------------------------------------------------------------
    # GCS-specific helpers
    # ------------------------------------------------------------------

    def get_gcsfs(self):
        """Return the underlying :class:`gcsfs.GCSFileSystem` instance."""
        return self._get_fs()

    def download_file(self, local_path: Path) -> None:
        """Download this GCS file to *local_path*."""
        fs = self._get_fs()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(self._raw(), str(local_path))

    def download_dir(self, local_path: Path) -> None:
        """Download this GCS directory's contents into *local_path*.

        The remote directory structure is preserved relative to *local_path*.
        For example ``gs://bucket/a/b/`` downloaded into ``/tmp/x`` produces
        ``/tmp/x/file1``, ``/tmp/x/sub/file2``, etc.
        """
        fs = self._get_fs()
        local_path.mkdir(parents=True, exist_ok=True)
        src = self._raw().rstrip("/")
        for remote_path in fs.find(src):
            rel = remote_path[len(src):].lstrip("/")
            if not rel:
                continue
            local_file = local_path / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            fs.get(remote_path, str(local_file))

    def get_pyarrow_filesystem(self):
        """Return a PyArrow-compatible filesystem wrapping gcsfs."""
        import pyarrow.fs as pafs  # noqa: PLC0415
        return pafs.PyFileSystem(pafs.FSSpecHandler(self.get_gcsfs()))

    # ------------------------------------------------------------------
    # Optional: make GCSPath usable as a dict key / in sets
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GCSPath):
            return self._uri == other._uri
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._uri)
