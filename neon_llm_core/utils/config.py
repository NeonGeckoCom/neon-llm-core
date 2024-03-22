# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
# BSD-3
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os

from dataclasses import dataclass
from os.path import join, dirname, isfile
from typing import Union

from neon_utils.log_utils import init_log
from ovos_utils.log import LOG
from ovos_config.config import Configuration

from neon_llm_core.utils.constants import LLM_VHOST
import neon_mq_connector.utils.client_utils as mq_connector_client_utils


def load_legacy_config() -> Union[dict, None]:
    legacy_config_path = os.getenv("NEON_LLM_LEGACY_CONFIG", "/app/app/config.json")
    if isfile(legacy_config_path):
        with open(legacy_config_path) as f:
            config = json.load(f)
        init_log(config=config)
        mq_connector_client_utils._default_mq_config = config.get("MQ")
        return config


load_ovos_config = Configuration


def load_default_config() -> Union[dict, None]:
    LOG.warning(f"No configuration found! falling back to defaults")
    default_config_path = join(dirname(__file__), "default_config.json")
    with open(default_config_path) as f:
        config = json.load(f)
    return config


def load_config() -> Union[dict, None]:
    """
    Load and return a configuration object,
    """
    configs_loading_order = (load_legacy_config, load_ovos_config, load_default_config,)
    for config_loader in configs_loading_order:
        config = config_loader()
        if config:
            LOG.info(f'Applied configs from loader={config_loader.__name__}()')
            return config


@dataclass
class LLMMQConfig:
    ask_response_queue: str
    ask_appraiser_queue: str
    ask_discusser_queue: str
    vhost: str = LLM_VHOST
