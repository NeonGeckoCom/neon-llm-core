# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2024 NeonGecko.com Inc.
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

import pytest

from unittest import TestCase
from unittest.mock import Mock

from mirakuru import ProcessExitedWithError
from port_for import get_port
from pytest_rabbitmq.factories.executor import RabbitMqExecutor
from pytest_rabbitmq.factories.process import get_config

from neon_llm_core.llm import NeonLLM
from neon_llm_core.rmq import NeonLLMMQConnector


class NeonMockLlm(NeonLLMMQConnector):
    def __init__(self):
        config = {"MQ": {"server": "127.0.0.1", "port": 25672,
                         "users": {
                             "mq_handler": {"user": "neon_api_utils",
                                            "password": "Klatchat2021"},
                             "neon_llm_mock_mq": {"user": "test_llm_user",
                                                  "password": "test_llm_password"}}}}
        NeonLLMMQConnector.__init__(self, config=config)

    @property
    def name(self):
        return "mock_mq"

    @property
    def model(self) -> NeonLLM:
        return Mock()

    @staticmethod
    def compose_opinion_prompt(respondent_nick: str,
                               question: str,
                               answer: str) -> str:
        return "opinion prompt"


@pytest.fixture(scope="class")
def rmq_instance(request, tmp_path_factory):
    config = get_config(request)
    rabbit_ctl = config["ctl"]
    rabbit_server = config["server"]
    rabbit_host = "127.0.0.1"
    rabbit_port = 25672
    rabbit_distribution_port = get_port(
        config["distribution_port"], [rabbit_port]
    )
    assert rabbit_distribution_port
    assert (
            rabbit_distribution_port != rabbit_port
    ), "rabbit_port and distribution_port can not be the same!"

    tmpdir = tmp_path_factory.mktemp(f"pytest-rabbitmq-{request.fixturename}")

    rabbit_plugin_path = config["plugindir"]

    rabbit_logpath = config["logsdir"]

    if not rabbit_logpath:
        rabbit_logpath = tmpdir / "logs"

    rabbit_executor = RabbitMqExecutor(
        rabbit_server,
        rabbit_host,
        rabbit_port,
        rabbit_distribution_port,
        rabbit_ctl,
        logpath=rabbit_logpath,
        path=tmpdir,
        plugin_path=rabbit_plugin_path,
        node_name=config["node"],
    )

    rabbit_executor.start()

    # Init RMQ config
    rabbit_executor.rabbitctl_output("add_user", "test_llm_user", "test_llm_password")
    rabbit_executor.rabbitctl_output("add_vhost", "/llm")
    rabbit_executor.rabbitctl_output("set_permissions_globally", "test_llm_user", ".*", ".*", ".*")

    request.cls.rmq_instance = rabbit_executor


@pytest.mark.usefixtures("rmq_instance")
class TestNeonLLMMQConnector(TestCase):

    @classmethod
    def tearDownClass(cls):
        try:
            cls.rmq_instance.stop()
        except ProcessExitedWithError:
            pass

    def test_00_init(self):
        self.mq_llm = NeonMockLlm()

        self.assertIn(self.mq_llm.name, self.mq_llm.service_name)
        self.assertIsInstance(self.mq_llm.ovos_config, dict)
        self.assertEqual(self.mq_llm.vhost, "/llm")
        self.assertIsNotNone(self.mq_llm.model)
        self.assertEqual(self.mq_llm._personas_provider.service_name,
                         self.mq_llm.name)
