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

import os
from unittest.mock import MagicMock, patch

import pytest

from neptune_exporter.storage.gcs import GCSPath, is_gcs_url


# ---------------------------------------------------------------------------
# is_gcs_url
# ---------------------------------------------------------------------------


def test_is_gcs_url_with_gcs_uri():
    assert is_gcs_url("gs://my-bucket/path/to/file") is True


def test_is_gcs_url_with_local_path():
    assert is_gcs_url("/local/path") is False


def test_is_gcs_url_with_relative_path():
    assert is_gcs_url("relative/path") is False


def test_is_gcs_url_with_empty_string():
    assert is_gcs_url("") is False


# ---------------------------------------------------------------------------
# GCSPath – pure property / operator tests (no filesystem needed)
# ---------------------------------------------------------------------------


class TestGCSPathProperties:
    def test_str(self):
        p = GCSPath("gs://bucket/a/b")
        assert str(p) == "gs://bucket/a/b"

    def test_repr(self):
        p = GCSPath("gs://bucket/a/b")
        assert repr(p) == "GCSPath('gs://bucket/a/b')"

    def test_fspath(self):
        p = GCSPath("gs://bucket/a/b")
        assert os.fspath(p) == "gs://bucket/a/b"

    def test_trailing_slash_stripped(self):
        p = GCSPath("gs://bucket/a/b/")
        assert str(p) == "gs://bucket/a/b"

    def test_name(self):
        assert GCSPath("gs://bucket/a/b/file.parquet").name == "file.parquet"

    def test_name_root(self):
        assert GCSPath("gs://bucket").name == "bucket"

    def test_stem(self):
        assert GCSPath("gs://bucket/file.parquet").stem == "file"

    def test_stem_no_extension(self):
        assert GCSPath("gs://bucket/file").stem == "file"

    def test_suffix(self):
        assert GCSPath("gs://bucket/file.parquet").suffix == ".parquet"

    def test_suffix_no_extension(self):
        assert GCSPath("gs://bucket/file").suffix == ""

    def test_parent(self):
        p = GCSPath("gs://bucket/a/b/c")
        assert str(p.parent) == "gs://bucket/a/b"

    def test_parent_top_level(self):
        # parent of "gs://bucket" should stay "gs://bucket" (no higher)
        p = GCSPath("gs://bucket")
        assert isinstance(p.parent, GCSPath)

    def test_gcs_path_strips_scheme(self):
        p = GCSPath("gs://bucket/a/b")
        assert p.gcs_path == "bucket/a/b"

    def test_truediv(self):
        p = GCSPath("gs://bucket/a")
        child = p / "b" / "c"
        assert str(child) == "gs://bucket/a/b/c"

    def test_truediv_strips_leading_slash(self):
        p = GCSPath("gs://bucket/a")
        child = p / "/b"
        assert str(child) == "gs://bucket/a/b"

    def test_absolute_returns_self(self):
        p = GCSPath("gs://bucket/a")
        assert p.absolute() is p

    def test_equality(self):
        assert GCSPath("gs://bucket/a") == GCSPath("gs://bucket/a")

    def test_inequality(self):
        assert GCSPath("gs://bucket/a") != GCSPath("gs://bucket/b")

    def test_hashable(self):
        s = {
            GCSPath("gs://bucket/a"),
            GCSPath("gs://bucket/a"),
            GCSPath("gs://bucket/b"),
        }
        assert len(s) == 2

    def test_equality_with_non_gcspath(self):
        assert GCSPath("gs://bucket/a").__eq__("/local/path") is NotImplemented


# ---------------------------------------------------------------------------
# GCSPath – lazy import error
# ---------------------------------------------------------------------------


def test_get_fs_raises_import_error_when_gcsfs_missing():
    p = GCSPath("gs://bucket/path")
    with patch.dict("sys.modules", {"gcsfs": None}):
        with pytest.raises(ImportError, match="gcsfs is required"):
            p._get_fs()


# ---------------------------------------------------------------------------
# GCSPath – filesystem operations (mocked gcsfs)
# ---------------------------------------------------------------------------


def _make_gcs_path(uri: str) -> tuple[GCSPath, MagicMock]:
    """Return a GCSPath pre-wired with a mock filesystem."""
    mock_fs = MagicMock()
    p = GCSPath(uri, _fs=mock_fs)
    return p, mock_fs


class TestGCSPathFsOps:
    def test_exists_true(self):
        p, fs = _make_gcs_path("gs://bucket/a/b")
        fs.exists.return_value = True
        assert p.exists() is True
        fs.exists.assert_called_once_with("bucket/a/b")

    def test_exists_false(self):
        p, fs = _make_gcs_path("gs://bucket/a/b")
        fs.exists.return_value = False
        assert p.exists() is False

    def test_is_dir_true(self):
        p, fs = _make_gcs_path("gs://bucket/a/b")
        fs.isdir.return_value = True
        assert p.is_dir() is True
        fs.isdir.assert_called_once_with("bucket/a/b")

    def test_iterdir(self):
        p, fs = _make_gcs_path("gs://bucket/dir")
        fs.ls.return_value = ["bucket/dir/file1.txt", "bucket/dir/subdir"]
        children = list(p.iterdir())
        assert len(children) == 2
        assert str(children[0]) == "gs://bucket/dir/file1.txt"
        assert str(children[1]) == "gs://bucket/dir/subdir"

    def test_iterdir_not_found(self):
        p, fs = _make_gcs_path("gs://bucket/missing")
        fs.ls.side_effect = FileNotFoundError
        assert list(p.iterdir()) == []

    def test_glob(self):
        p, fs = _make_gcs_path("gs://bucket/dir")
        fs.glob.return_value = ["bucket/dir/a.parquet", "bucket/dir/b.parquet"]
        results = list(p.glob("*.parquet"))
        assert len(results) == 2
        assert str(results[0]) == "gs://bucket/dir/a.parquet"
        fs.glob.assert_called_once_with("bucket/dir/*.parquet")

    def test_get_gcsfs(self):
        p, fs = _make_gcs_path("gs://bucket/a")
        assert p.get_gcsfs() is fs


# ---------------------------------------------------------------------------
# GCSPath – download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def test_download_file(self, tmp_path):
        p, fs = _make_gcs_path("gs://bucket/dir/file.txt")
        local = tmp_path / "file.txt"
        p.download_file(local)
        fs.get.assert_called_once_with("bucket/dir/file.txt", str(local))

    def test_download_file_creates_parent_dirs(self, tmp_path):
        p, fs = _make_gcs_path("gs://bucket/dir/file.txt")
        local = tmp_path / "nested" / "dirs" / "file.txt"
        p.download_file(local)
        assert local.parent.exists()


# ---------------------------------------------------------------------------
# GCSPath – download_dir
# ---------------------------------------------------------------------------


class TestDownloadDir:
    def test_download_dir_preserves_structure(self, tmp_path):
        p, fs = _make_gcs_path("gs://bucket/exports")
        fs.find.return_value = [
            "bucket/exports/file1.txt",
            "bucket/exports/sub/file2.txt",
        ]

        p.download_dir(tmp_path)

        assert fs.get.call_count == 2
        fs.get.assert_any_call("bucket/exports/file1.txt", str(tmp_path / "file1.txt"))
        fs.get.assert_any_call(
            "bucket/exports/sub/file2.txt", str(tmp_path / "sub" / "file2.txt")
        )

    def test_download_dir_skips_root_entry(self, tmp_path):
        """An entry equal to the source directory itself should be skipped."""
        p, fs = _make_gcs_path("gs://bucket/exports")
        fs.find.return_value = ["bucket/exports"]  # root entry only

        p.download_dir(tmp_path)

        fs.get.assert_not_called()

    def test_download_dir_creates_local_dir(self, tmp_path):
        p, fs = _make_gcs_path("gs://bucket/exports")
        fs.find.return_value = []
        target = tmp_path / "new_dir"
        p.download_dir(target)
        assert target.exists()


# ---------------------------------------------------------------------------
# GCSPath – get_pyarrow_filesystem
# ---------------------------------------------------------------------------


def test_get_pyarrow_filesystem():
    p, mock_fs = _make_gcs_path("gs://bucket/a")

    mock_handler = MagicMock()
    mock_pa_fs = MagicMock()

    with (
        patch(
            "pyarrow.fs.FSSpecHandler", return_value=mock_handler
        ) as mock_handler_cls,
        patch("pyarrow.fs.PyFileSystem", return_value=mock_pa_fs) as mock_pyfs_cls,
    ):
        result = p.get_pyarrow_filesystem()

    mock_handler_cls.assert_called_once_with(mock_fs)
    mock_pyfs_cls.assert_called_once_with(mock_handler)
    assert result is mock_pa_fs


# ---------------------------------------------------------------------------
# GCSPath – lazy _fs initialisation via gcsfs
# ---------------------------------------------------------------------------


def test_fs_is_lazily_initialised():
    mock_fs_instance = MagicMock()
    mock_gcsfs_module = MagicMock()
    mock_gcsfs_module.GCSFileSystem.return_value = mock_fs_instance

    p = GCSPath("gs://bucket/a")  # no _fs provided
    assert p._fs is None

    with patch.dict("sys.modules", {"gcsfs": mock_gcsfs_module}):
        fs = p._get_fs()

    assert fs is mock_fs_instance
    # Second call should reuse the cached instance
    assert p._get_fs() is mock_fs_instance
    mock_gcsfs_module.GCSFileSystem.assert_called_once()
