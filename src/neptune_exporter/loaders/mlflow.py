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

from typing import List, Optional


class MLflowLoader:
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri

    def create_project(self, project: str) -> None:
        pass

    def create_run(self, project: str, run: str) -> None:
        pass

    def upload_metrics(self, metrics: List[str], run_id: str) -> None:
        pass

    def upload_configs(self, parameters: List[str], run_id: str) -> None:
        pass

    def upload_artifacts(self, artifacts: List[str], run_id: str) -> None:
        pass
