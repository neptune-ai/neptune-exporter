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


from neptune_exporter.exporters.neptune3 import Neptune3Exporter
from neptune_exporter.manager import ExportManager
from neptune_exporter.storage.parquet import ParquetStorage
from pathlib import Path


def main() -> None:
    export_manager = ExportManager(
        exporter=Neptune3Exporter(),
        storage=ParquetStorage(base_path=Path("./examples")),
    )
    export_manager.run(
        project_ids=["examples/LLM-Pretraining"],
    )


if __name__ == "__main__":
    main()
