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
from pytest_rabbitmq.factories.executor import RabbitMqExecutor
from neon_minerva.integration.rabbit_mq import rmq_instance
from neon_data_models.models.api.mq import LLMProposeResponse, LLMDiscussResponse
from neon_data_models.models.api.llm import LLMPersona, LLMRequest

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
        self._model = Mock(NeonLLM)
        self._model.llm_model_name = "mock_llm@test"
        self._model.ask.return_value = "Mock response"
        self._model.ask_discusser.return_value = LLMDiscussResponse(
            opinion="Mock opinion")
        self._model.query_model.return_value = LLMProposeResponse(
            response="Mock response")
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
                                    persona=LLMPersona(persona_name="vanilla",
                                                       enabled=True),
                                    model=self.mq_llm.model.llm_model_name,
                                    query="Mock Query", history=[])
        self.mq_llm.handle_request(None, None, None,
                                   dict_to_b64(request.model_dump())).join()
        self.mq_llm.model.query_model.assert_called_with(
            LLMRequest(**request.model_dump()))

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMProposeResponse(**response['request_data'])
        self.assertIsInstance(response, LLMProposeResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)

        self.assertEqual(response.response,
                          self.mq_llm.model.query_model(LLMProposeRequest(
                              **request.model_dump())).response)

    def test_handle_opinion_request(self):
        from neon_data_models.models.api.mq import (LLMDiscussRequest,
                                                    LLMDiscussResponse)
        # Valid Request
        request = LLMDiscussRequest(message_id="mock_message_id",
                                    routing_key="mock_routing_key",
                                    query="Mock Discuss", history=[],
                                    options={"bot 1": "resp 1",
                                             "bot 2": "resp 2"})
        # Mock the ask_discusser method to return a known response
        discuss_response = LLMDiscussResponse(message_id=request.message_id,
                                               routing_key=request.routing_key,
                                               opinion="Mock opinion")
        self.mq_llm.model.ask_discusser.return_value = discuss_response

        self.mq_llm.handle_opinion_request(None, None, None,
                                           dict_to_b64(request.model_dump())).join()

        # Verify ask_discusser was called with the right parameters
        self.mq_llm.model.ask_discusser.assert_called_once()

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMDiscussResponse(**response['request_data'])
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)
        self.assertEqual(response.opinion, "Mock opinion")

        # No input options
        request = LLMDiscussRequest(message_id="mock_message_id1",
                                    routing_key="mock_routing_key1",
                                    query="Mock Discuss 1", history=[],
                                    options={})
        # Mock a different response for the empty options case
        empty_discuss_response = LLMDiscussResponse(message_id=request.message_id,
                                                    routing_key=request.routing_key,
                                                    opinion="Sorry, but I got no options to choose from.")
        self.mq_llm.model.ask_discusser.return_value = empty_discuss_response

        self.mq_llm.handle_opinion_request(None, None, None,
                                           dict_to_b64(request.model_dump())).join()

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMDiscussResponse(**response['request_data'])
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)
        self.assertEqual(response.opinion, "Sorry, but I got no options to choose from.")

    def test_handle_score_request(self):
        from neon_data_models.models.api.mq import (LLMVoteRequest,
                                                    LLMVoteResponse)

        # Valid Request
        request = LLMVoteRequest(message_id="mock_message_id",
                                 routing_key="mock_routing_key",
                                 query="Mock Score", history=[],
                                 responses=["one", "two"])

        # Mock the ask_appraiser method to return a known response
        vote_response = LLMVoteResponse(message_id=request.message_id,
                                        routing_key=request.routing_key,
                                        sorted_answer_indexes=[0, 1])
        self.mq_llm.model.ask_appraiser.return_value = vote_response

        self.mq_llm.handle_score_request(None, None, None,
                                         dict_to_b64(request.model_dump())).join()

        # Verify ask_appraiser was called with the right parameters
        self.mq_llm.model.ask_appraiser.assert_called_once()

        response = self.mq_llm.send_message.call_args.kwargs
        self.assertEqual(response['queue'], request.routing_key)
        response = LLMVoteResponse(**response['request_data'])
        self.assertIsInstance(response, LLMVoteResponse)
        self.assertEqual(request.routing_key, response.routing_key)
        self.assertEqual(request.message_id, response.message_id)
        self.assertEqual(response.sorted_answer_indexes, [0, 1])
