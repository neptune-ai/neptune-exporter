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

"""Loaders package for uploading data to target platforms."""

from .loader import DataLoader

try:
    from .mlflow_loader import MLflowLoader

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - mlflow and MLflowLoader are optional
    MLflowLoader = None  # type: ignore[misc,assignment]
    MLFLOW_AVAILABLE = False  # type: ignore[misc,assignment]

try:
    from .wandb_loader import WandBLoader

    WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - wandb and WandBLoader are optional
    WandBLoader = None  # type: ignore[misc,assignment]
    WANDB_AVAILABLE = False  # type: ignore[misc,assignment]

try:
    from .comet_loader import CometLoader

    COMET_AVAILABLE = True
except Exception:  # pragma: no cover - comet-ml and CometLoader are optional
    CometLoader = None  # type: ignore[misc,assignment]
    COMET_AVAILABLE = False  # type: ignore[misc,assignment]

try:
    from .zenml_loader import ZenMLLoader, ZENML_AVAILABLE
except Exception:  # pragma: no cover - zenml and ZenMLLoader are optional
    ZenMLLoader = None  # type: ignore[misc,assignment]
    ZENML_AVAILABLE = False  # type: ignore[misc,assignment]

try:
    from .litlogger_loader import LitLoggerLoader, LITLOGGER_AVAILABLE
except Exception:  # pragma: no cover - litlogger and LitLoggerLoader are optional
    LitLoggerLoader = None  # type: ignore[misc,assignment]
    LITLOGGER_AVAILABLE = False  # type: ignore[misc,assignment]

try:
    from . import minfx_loader
    from .minfx_loader import MinfxLoader

    MINFX_AVAILABLE = True
except Exception:  # pragma: no cover - minfx is optional
    minfx_loader = None  # type: ignore[misc,assignment]
    MinfxLoader = None  # type: ignore[misc,assignment]
    MINFX_AVAILABLE = False  # type: ignore[misc,assignment]

__all__ = ["DataLoader"]
if MLFLOW_AVAILABLE:
    __all__.append("MLflowLoader")
if WANDB_AVAILABLE:
    __all__.append("WandBLoader")
if COMET_AVAILABLE:
    __all__.append("CometLoader")
if LITLOGGER_AVAILABLE:
    __all__.append("LitLoggerLoader")
if ZENML_AVAILABLE:
    __all__.append("ZenMLLoader")
if MINFX_AVAILABLE:
    __all__.append("MinfxLoader")
