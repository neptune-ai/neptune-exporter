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
import sys
from typing import TextIO

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes via tqdm to avoid progress bar corruption."""

    def __init__(self, level: int = logging.NOTSET, file: TextIO | None = None) -> None:
        super().__init__(level)
        self._file = file if file is not None else sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self._file)
        except Exception:
            self.handleError(record)


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
