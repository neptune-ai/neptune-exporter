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

import logging
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from rich.console import Console as RichConsole
    from rich.highlighter import Highlighter as RichHighlighter
    from rich.theme import Theme as RichTheme
else:  # pragma: no cover - typing only
    RichConsole = Any
    RichHighlighter = Any
    RichTheme = Any

_Console: type[RichConsole] | None = None
_RichHandler: Any = None
_Theme: type[RichTheme] | None = None
_NullHighlighter: type[RichHighlighter] | None = None
_RICH_AVAILABLE = False

try:
    from rich.console import Console as _Console
    from rich.highlighter import NullHighlighter as _NullHighlighter
    from rich.logging import RichHandler as _RichHandler
    from rich.theme import Theme as _Theme

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - rich is an optional import at runtime
    _RICH_AVAILABLE = False

_RICH_CONSOLE: "RichConsole | None" = None
_RICH_THEME: "RichTheme | None" = None

_RICH_STYLES: dict[str, str] = {
    # log
    "log.time": "#8D9195",
    "log.level": "bold #5B69C2",
    "log.message": "#FAFBFB",
    "log.path": "#5B5F63",
    # logging levels
    "logging.level.debug": "#5B5F63",
    "logging.level.info": "#65C4EA",
    "logging.level.warning": "#DFA045",
    "logging.level.error": "#EA7987",
    "logging.level.critical": "bold #EA7987",
    # progress
    "progress.spinner": "#5B69C2",
    "progress.description": "#FAFBFB",
    "progress.download": "#BABEC3",
    "bar.back": "#2F3132",
    # differentiate: in-progress vs complete
    "bar.complete": "#9AA4E7",  # in-progress fill (lighter)
    "bar.pulse": "#65C4EA",  # animated pulse highlight
    "bar.finished": "#5B69C2",  # completed fill (primary)
    "progress.percentage": "#9AA4E7",
    "progress.elapsed": "#8D9195",
}


class ConsoleLevelFilter(logging.Filter):
    """Filter records for console output based on desired stderr level."""

    def __init__(self, stderr_level: int) -> None:
        super().__init__()
        self._stderr_level = stderr_level

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "always_console", False):
            return True
        return record.levelno >= self._stderr_level


def info_always(logger: logging.Logger, message: str) -> None:
    logger.info(message, extra={"always_console": True})


def get_rich_console() -> "RichConsole":
    if not _RICH_AVAILABLE or _Console is None:
        raise RuntimeError("rich is required for rich console output")
    global _RICH_CONSOLE
    if _RICH_CONSOLE is None:
        _RICH_CONSOLE = _Console(stderr=True, theme=_get_rich_theme())
    return _RICH_CONSOLE


def _get_rich_theme() -> "RichTheme":
    if not _RICH_AVAILABLE or _Theme is None:
        raise RuntimeError("rich is required for rich console output")
    global _RICH_THEME
    if _RICH_THEME is None:
        _RICH_THEME = _Theme(_RICH_STYLES)
    return _RICH_THEME


def create_console_handler(stderr_level: int) -> logging.Handler:
    if not _RICH_AVAILABLE or _RichHandler is None or _NullHighlighter is None:
        raise RuntimeError("rich is required for rich logging output")
    handler: logging.Handler = cast(Any, _RichHandler)(
        console=get_rich_console(),
        show_time=True,
        show_level=True,
        show_path=False,
        highlighter=_NullHighlighter(),
        keywords=[],
        markup=False,
        rich_tracebacks=True,
    )
    handler.setLevel(logging.INFO)
    handler.addFilter(ConsoleLevelFilter(stderr_level))
    return handler
