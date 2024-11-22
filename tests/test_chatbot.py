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

from unittest import TestCase
from unittest.mock import patch

from neon_data_models.models.api import LLMPersona, LLMProposeRequest, LLMProposeResponse, LLMDiscussRequest, \
    LLMDiscussResponse, LLMVoteRequest, LLMVoteResponse
from pydantic import ValidationError

from neon_llm_core.chatbot import LLMBot
from neon_llm_core.utils.config import LLMMQConfig


class MockChatbot(LLMBot):
    def __init__(self):
        LLMBot.__init__(self, llm_name="mock_chatbot",
                        persona={"name": "test_persona",
                                 "system_prompt": "Test Prompt"})


class TestChatbot(TestCase):
    mock_chatbot = MockChatbot()

    @classmethod
    def tearDownClass(cls):
        cls.mock_chatbot.shutdown()

    def test_00_init(self):
        self.assertEqual(self.mock_chatbot.bot_type, "submind")
        self.assertIsInstance(self.mock_chatbot.base_llm, str)
        self.assertIsInstance(self.mock_chatbot.persona, LLMPersona)
        self.assertIsInstance(self.mock_chatbot.mq_queue_config, LLMMQConfig)

    def test_ask_chatbot(self):
        # TODO
        pass

    def test_ask_discusser(self):
        # TODO
        pass

    def test_ask_appraiser(self):
        # TODO
        pass

    @patch('neon_llm_core.chatbot.send_mq_request')
    def test_get_llm_api_response(self, mq_request):
        mq_request.return_value = {"response": "test",
                                   "message_id": ""}

        # Valid Request
        resp = self.mock_chatbot._get_llm_api_response("input")
        request_data = mq_request.call_args.kwargs['request_data']
        req = LLMProposeRequest(**request_data)
        self.assertIsInstance(req, LLMProposeRequest)
        self.assertEqual(req.query, "input")
        self.assertEqual(req.model, self.mock_chatbot.base_llm)
        self.assertEqual(req.persona, self.mock_chatbot.persona)
        self.assertIsInstance(resp, LLMProposeResponse)
        self.assertEqual(resp.response, mq_request.return_value['response'])

        # Invalid request
        self.assertIsNone(self.mock_chatbot._get_llm_api_response(None))

        # Invalid response
        mq_request.return_value = {}
        self.assertIsNone(self.mock_chatbot._get_llm_api_response("input"))

    @patch('neon_llm_core.chatbot.send_mq_request')
    def test_get_llm_api_opinion(self, mq_request):
        mq_request.return_value = {"opinion": "test",
                                   "message_id": ""}
        prompt = "test prompt"
        options = {"bot 1": "resp 1", "bot 2": "resp 2"}

        # Valid Request
        resp = self.mock_chatbot._get_llm_api_opinion(prompt, options)
        request_data = mq_request.call_args.kwargs['request_data']
        req = LLMDiscussRequest(**request_data)
        self.assertIsInstance(req, LLMDiscussRequest)
        self.assertEqual(req.query, prompt)
        self.assertEqual(req.options, options)
        self.assertEqual(req.model, self.mock_chatbot.base_llm)
        self.assertEqual(req.persona, self.mock_chatbot.persona)
        self.assertIsInstance(resp, LLMDiscussResponse)
        self.assertEqual(resp.opinion, mq_request.return_value['opinion'])

        # Invalid request
        self.assertIsNone(self.mock_chatbot._get_llm_api_opinion(prompt,
                                                                 prompt))

        # Invalid response
        mq_request.return_value = {}
        self.assertIsNone(self.mock_chatbot._get_llm_api_opinion(prompt,
                                                                 options))

    @patch('neon_llm_core.chatbot.send_mq_request')
    def test_get_llm_api_choice(self, mq_request):
        mq_request.return_value = {"sorted_answer_indexes": [2, 0, 1],
                                   "message_id": ""}
        prompt = "test prompt"
        responses = ["one", "two", "three"]

        # Valid Request
        resp = self.mock_chatbot._get_llm_api_choice(prompt, responses)
        request_data = mq_request.call_args.kwargs['request_data']

        req = LLMVoteRequest(**request_data)
        self.assertIsInstance(req, LLMVoteRequest)
        self.assertEqual(req.query, prompt)
        self.assertEqual(req.responses, responses)
        self.assertEqual(req.model, self.mock_chatbot.base_llm)
        self.assertEqual(req.persona, self.mock_chatbot.persona)
        self.assertIsInstance(resp, LLMVoteResponse)
        self.assertEqual(resp.sorted_answer_indexes,
                         mq_request.return_value['sorted_answer_indexes'])

        # Invalid request
        self.assertIsNone(self.mock_chatbot._get_llm_api_choice(prompt,
                                                                [1, 2, 3]))

        # Invalid response
        mq_request.return_value["sorted_answer_indexes"] = ["one", "two",
                                                            "three"]
        self.assertIsNone(self.mock_chatbot._get_llm_api_choice(prompt,
                                                                responses))
