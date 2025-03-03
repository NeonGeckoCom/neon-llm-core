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
from datetime import datetime
from unittest import TestCase
from unittest.mock import patch

from neon_data_models.models.api import (
    LLMPersona,
    LLMProposeRequest,
    LLMProposeResponse,
    LLMDiscussRequest,
    LLMDiscussResponse,
    LLMVoteRequest,
    LLMVoteResponse,
)

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

    @patch.object(mock_chatbot, '_get_llm_api_response')
    def test_ask_chatbot(self, get_api_response):
        get_api_response.return_value = LLMProposeResponse(message_id="",
                                                           response="test_resp")
        valid_prompt_id = "test_prompt_id_ask"
        valid_user = "test_user"
        valid_shout = "test_shout"
        valid_timestamp = datetime.now().isoformat()
        valid_context = {"prompt_id": valid_prompt_id}

        # Valid Request
        resp = self.mock_chatbot.ask_chatbot(valid_user, valid_shout,
                                             valid_timestamp, valid_context)
        get_api_response.assert_called_with(shout=valid_shout)
        self.assertEqual(resp, "test_resp")
        self.assertEqual(self.mock_chatbot.prompt_id_to_shout[valid_prompt_id],
                         valid_shout)

        # Valid without context
        resp = self.mock_chatbot.ask_chatbot(valid_user, valid_shout,
                                             valid_timestamp)
        get_api_response.assert_called_with(shout=valid_shout)
        self.assertEqual(resp, "test_resp")
        self.assertEqual(self.mock_chatbot.prompt_id_to_shout[valid_prompt_id],
                         valid_shout)

        # Invalid request
        self.assertIsInstance(self.mock_chatbot.ask_chatbot(valid_user,
                                                            None,
                                                            valid_timestamp),
                              str)
        get_api_response.assert_called_with(shout=None)

        # Invalid response
        get_api_response.return_value = None
        from neon_llm_core.utils.constants import DEFAULT_RESPONSE
        self.assertEqual(self.mock_chatbot.ask_chatbot(valid_user,
                                                       valid_shout,
                                                       valid_timestamp,
                                                       valid_context),
                         DEFAULT_RESPONSE)
        get_api_response.assert_called_with(shout=valid_shout)

    @patch.object(mock_chatbot, '_get_llm_api_opinion')
    def test_ask_discusser(self, get_api_opinion):
        get_api_opinion.return_value = LLMDiscussResponse(message_id="",
                                                          opinion="test_resp")
        valid_prompt_id = "test_prompt_id_disc"
        valid_prompt = "test prompt"
        valid_options = {"bot 1": "response 1", "bot 2": "response 2"}
        valid_context = {"prompt_id": valid_prompt_id}

        self.mock_chatbot.prompt_id_to_shout[valid_prompt_id] = valid_prompt

        # Valid request
        resp = self.mock_chatbot.ask_discusser(valid_options, valid_context)
        get_api_opinion.assert_called_with(prompt=valid_prompt,
                                           options=valid_options)
        self.assertEqual(resp, "test_resp")

        # Invalid response
        get_api_opinion.return_value = None
        from neon_llm_core.utils.constants import DEFAULT_RESPONSE
        self.assertEqual(self.mock_chatbot.ask_discusser(valid_options,
                                                         valid_context),
                         DEFAULT_RESPONSE)
        get_api_opinion.assert_called_with(prompt=valid_prompt,
                                           options=valid_options)

    @patch.object(mock_chatbot, '_get_llm_api_choice')
    def test_ask_appraiser(self, get_api_choice):
        get_api_choice.return_value = LLMVoteResponse(
            message_id="", sorted_answer_indexes=[2, 0, 1])
        valid_prompt_id = "test_prompt_id_vote"
        valid_prompt = "test prompt"
        options = {"bot 0": "response 0",
                   "bot 1": "response 1",
                   "bot 2": "response 2",
                   self.mock_chatbot.service_name: "Self response"}
        valid_options = ["response 0", "response 1", "response 2"]
        valid_context = {"prompt_id": valid_prompt_id}

        self.mock_chatbot.prompt_id_to_shout[valid_prompt_id] = valid_prompt

        # Valid request
        resp = self.mock_chatbot.ask_appraiser(options, valid_context)
        get_api_choice.assert_called_with(prompt=valid_prompt,
                                          responses=valid_options)
        self.assertEqual(resp, "bot 2")

        # Invalid no valid options
        resp = self.mock_chatbot.ask_appraiser(
            {self.mock_chatbot.service_name: "Self response"},
            valid_context)
        self.assertIn("abstain", resp.lower())

        # Invalid API response
        get_api_choice.reset_mock()
        get_api_choice.return_value = None
        resp = self.mock_chatbot.ask_appraiser(options, valid_context)
        get_api_choice.assert_called_with(prompt=valid_prompt,
                                          responses=valid_options)
        self.assertIn("abstain", resp.lower())

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
