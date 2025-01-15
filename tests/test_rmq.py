# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 NeonGecko.com Inc.
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
from neon_mq_connector.consumers import SelectConsumerThread
from neon_mq_connector.utils.network_utils import dict_to_b64
from port_for import get_port
from pytest_rabbitmq.factories.executor import RabbitMqExecutor
from pytest_rabbitmq.factories.process import get_config

from neon_llm_core.llm import NeonLLM
from neon_llm_core.rmq import NeonLLMMQConnector


class NeonMockLlm(NeonLLMMQConnector):
    def __init__(self, rmq_port: int):
        config = {"MQ": {"server": "127.0.0.1", "port": rmq_port,
                         "users": {
                             "mq_handler": {"user": "neon_api_utils",
                                            "password": "Klatchat2021"},
                             "neon_llm_mock_mq": {"user": "test_llm_user",
                                                  "password": "test_llm_password"}}}}
        NeonLLMMQConnector.__init__(self, config=config)
        self._model = Mock()
        self._model.ask.return_value = "Mock response"
        self._model.get_sorted_answer_indexes.return_value = [0, 1]
        self.send_message = Mock()
        self._compose_opinion_prompt = Mock(return_value="Mock opinion prompt")

    @property
    def name(self):
        return "mock_mq"

    @property
    def model(self) -> NeonLLM:
        return self._model

    def compose_opinion_prompt(self, respondent_nick: str,
                               question: str,
                               answer: str) -> str:
        return self._compose_opinion_prompt(respondent_nick, question, answer)


@pytest.fixture(scope="class")
def rmq_instance(request, tmp_path_factory):
    config = get_config(request)
    rabbit_ctl = config["ctl"]
    rabbit_server = config["server"]
    rabbit_host = "127.0.0.1"
    rabbit_port = get_port(config["port"])
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
    rabbit_executor.rabbitctl_output("add_user", "test_llm_user",
                                     "test_llm_password")
    rabbit_executor.rabbitctl_output("add_vhost", "/llm")
    rabbit_executor.rabbitctl_output("set_permissions", "-p", "/llm",
                                     "test_llm_user", ".*", ".*", ".*")
    request.cls.rmq_instance = rabbit_executor


@pytest.mark.usefixtures("rmq_instance")
class TestNeonLLMMQConnector(TestCase):
    mq_llm: NeonMockLlm = None
    rmq_instance: RabbitMqExecutor = None

    @classmethod
    def tearDownClass(cls):
        try:
            cls.rmq_instance.stop()
        except ProcessExitedWithError:
            pass

    def setUp(self):
        if self.mq_llm is None:
            self.mq_llm = NeonMockLlm(self.rmq_instance.port)

    def test_00_init(self):
        self.assertIn(self.mq_llm.name, self.mq_llm.service_name)
        self.assertIsInstance(self.mq_llm.ovos_config, dict)
        self.assertEqual(self.mq_llm.vhost, "/llm")
        self.assertIsNotNone(self.mq_llm.model, self.mq_llm.model)
        self.assertEqual(self.mq_llm._personas_provider.service_name,
                         self.mq_llm.name)
        self.assertTrue(self.mq_llm.async_consumers_enabled)
        self.assertEqual(self.mq_llm.consumer_thread_cls, SelectConsumerThread)
        for consumer in self.mq_llm.consumers.values():
            self.assertIsInstance(consumer, SelectConsumerThread)

    def test_handle_request(self):
        from neon_data_models.models.api.mq import (LLMProposeRequest,
                                                    LLMProposeResponse)
        # Valid Request
        request = LLMProposeRequest(message_id="mock_message_id",
                                    routing_key="mock_routing_key",
                                    query="Mock Query", history=[])
        self.mq_llm.handle_request(None, None, None,
                                   dict_to_b64(request.model_dump())).join()
        self.mq_llm.model.ask.assert_called_with(message=request.query,
                                                 chat_history=request.history,
                                                 persona=request.persona)
        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMProposeResponse(**response['request_data'])
        self.assertIsInstance(response, LLMProposeResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)

        self.assertEqual(response.response, self.mq_llm.model.ask())

    def test_handle_opinion_request(self):
        from neon_data_models.models.api.mq import (LLMDiscussRequest,
                                                    LLMDiscussResponse)
        # Valid Request
        request = LLMDiscussRequest(message_id="mock_message_id",
                                    routing_key="mock_routing_key",
                                    query="Mock Discuss", history=[],
                                    options={"bot 1": "resp 1",
                                             "bot 2": "resp 2"})
        self.mq_llm.handle_opinion_request(None, None, None,
                                           dict_to_b64(request.model_dump()))

        self.mq_llm._compose_opinion_prompt.assert_called_with(
            list(request.options.keys())[0], request.query,
            list(request.options.values())[0])

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMDiscussResponse(**response['request_data'])
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)

        self.assertEqual(response.opinion, self.mq_llm.model.ask())

        # No input options
        request = LLMDiscussRequest(message_id="mock_message_id1",
                                    routing_key="mock_routing_key1",
                                    query="Mock Discuss 1", history=[],
                                    options={})
        self.mq_llm.handle_opinion_request(None, None, None,
                                           dict_to_b64(request.model_dump()))
        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMDiscussResponse(**response['request_data'])
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)
        self.assertNotEqual(response.opinion, self.mq_llm.model.ask())

        # TODO: Test with invalid sorted answer indexes

    def test_handle_score_request(self):
        from neon_data_models.models.api.mq import (LLMVoteRequest,
                                                    LLMVoteResponse)

        # Valid Request
        request = LLMVoteRequest(message_id="mock_message_id",
                                 routing_key="mock_routing_key",
                                 query="Mock Score", history=[],
                                 responses=["one", "two"])
        self.mq_llm.handle_score_request(None, None, None,
                                         dict_to_b64(request.model_dump()))

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMVoteResponse(**response['request_data'])
        self.assertIsInstance(response, LLMVoteResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)

        self.assertEqual(response.sorted_answer_indexes,
                         self.mq_llm.model.get_sorted_answer_indexes())

        # No response options
        request = LLMVoteRequest(message_id="mock_message_id",
                                 routing_key="mock_routing_key",
                                 query="Mock Score", history=[], responses=[])
        self.mq_llm.handle_score_request(None, None, None,
                                         dict_to_b64(request.model_dump()))

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMVoteResponse(**response['request_data'])
        self.assertIsInstance(response, LLMVoteResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)

        self.assertEqual(response.sorted_answer_indexes, [])
