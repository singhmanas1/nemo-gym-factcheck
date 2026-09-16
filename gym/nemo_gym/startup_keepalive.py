# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import threading
from time import sleep
from typing import Callable, List, Optional, Tuple

import requests
from omegaconf import DictConfig

from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.server_utils import ServerStatus

NEMO_GYM_STARTUP_KEEPALIVE_ENV = "NEMO_GYM_STARTUP_KEEPALIVE"
NEMO_GYM_STARTUP_KEEPALIVE_INTERVAL_ENV = "NEMO_GYM_STARTUP_KEEPALIVE_INTERVAL_S"
DEFAULT_KEEPALIVE_INTERVAL_S = 15


def startup_keepalive_enabled() -> bool:
    value = os.environ.get(NEMO_GYM_STARTUP_KEEPALIVE_ENV, "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def startup_keepalive_interval_s() -> float:
    raw = os.environ.get(
        NEMO_GYM_STARTUP_KEEPALIVE_INTERVAL_ENV, str(DEFAULT_KEEPALIVE_INTERVAL_S)
    )
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(DEFAULT_KEEPALIVE_INTERVAL_S)


def _server_should_keepalive(global_config_dict: DictConfig, server_name: str) -> bool:
    try:
        server_config_dict = get_first_server_config_dict(global_config_dict, server_name)
    except Exception:
        return False
    return bool(server_config_dict.get("spinup_server"))


def ping_spinup_server(
    global_config_dict: DictConfig,
    server_name: str,
    *,
    timeout_s: float = 120.0,
) -> None:
    """Send a minimal inference request to keep GPU-backed spinup servers active."""
    if not _server_should_keepalive(global_config_dict, server_name):
        return

    server_config_dict = get_first_server_config_dict(global_config_dict, server_name)
    host = server_config_dict.get("host")
    port = server_config_dict.get("port")
    if not host or not port:
        return

    base_url = f"http://{host}:{port}"
    model = server_config_dict.get("model")
    if not model:
        return

    requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        },
        timeout=timeout_s,
    )


def ping_ready_spinup_servers(
    global_config_dict: DictConfig,
    statuses: List[Tuple[str, ServerStatus]],
) -> None:
    for server_name, status in statuses:
        if status != "success":
            continue
        try:
            ping_spinup_server(global_config_dict, server_name)
        except Exception:
            # Fire-and-forget: startup keepalive must never block or fail spinup.
            pass


class StartupKeepalive:
    def __init__(
        self,
        get_statuses: Callable[[], List[Tuple[str, ServerStatus]]],
        *,
        interval_s: Optional[float] = None,
    ) -> None:
        self._get_statuses = get_statuses
        self._interval_s = (
            startup_keepalive_interval_s() if interval_s is None else interval_s
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not startup_keepalive_enabled():
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-gym-startup-keepalive",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            global_config_dict = get_global_config_dict()
            ping_ready_spinup_servers(global_config_dict, self._get_statuses())
            self._stop_event.wait(self._interval_s)
