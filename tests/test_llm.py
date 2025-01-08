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
from unittest.mock import Mock
from neon_data_models.models.api import LLMResponse

from neon_llm_core.llm import NeonLLM


class MockLLM(NeonLLM):

    mq_to_llm_role = {"user": "user",
                      "llm": "assistant"}

    def __init__(self, *args, **kwargs):
        NeonLLM.__init__(self, *args, **kwargs)
        self._assemble_prompt = Mock(return_value=lambda *args: args[0])
        self._tokenize = Mock(return_value=lambda *args: args[0])
        self.get_sorted_answer_indexes = Mock(return_value=lambda *args: [i for i in range(len(args))])
        self._call_model = Mock(return_value="mock model response")

    @property
    def tokenizer(self):
        return None

    @property
    def tokenizer_model_name(self) -> str:
        return "mock_tokenizer"

    @property
    def model(self):
        return Mock()

    @property
    def llm_model_name(self) -> str:
        return "mock_model"

    @property
    def _system_prompt(self) -> str:
        return "mock system prompt"


class TestNeonLLM(TestCase):
    MockLLM.__abstractmethods__ = set()
    config = {"test_config": True}
    test_llm = MockLLM(config)

    def setUp(self):
        self.test_llm._assemble_prompt.reset_mock()
        self.test_llm._tokenize.reset_mock()
        self.test_llm.get_sorted_answer_indexes.reset_mock()
        self.test_llm._call_model.reset_mock()

    def test_init(self):
        self.assertEqual(self.test_llm.llm_config, self.config)
        self.assertIsNone(self.test_llm.tokenizer)
        self.assertIsInstance(self.test_llm.tokenizer_model_name, str)
        self.assertIsNotNone(self.test_llm.model)
        self.assertIsInstance(self.test_llm.llm_model_name, str)
        self.assertIsInstance(self.test_llm._system_prompt, str)

    def test_ask(self):
        from neon_data_models.models.api import LLMPersona
        message = "Test input"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")

        # Valid request
        response = self.test_llm.ask(message, history, persona.model_dump())
        self.assertEqual(response, self.test_llm._call_model.return_value)
        self.test_llm._assemble_prompt.assert_called_once_with(message, history,
                                                               persona.model_dump())
        self.test_llm._call_model.assert_called_once_with(self.test_llm._assemble_prompt.return_value)

    def test_query_model(self):
        from neon_data_models.models.api import LLMPersona, LLMRequest
        message = "Test input"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")
        valid_request = LLMRequest(query=message, history=history,
                                   persona=persona,
                                   model=self.test_llm.llm_model_name)
        response = self.test_llm.query_model(valid_request)
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.response, self.test_llm._call_model.return_value)
        self.assertEqual(response.history[-1],
                         ("llm", self.test_llm._call_model.return_value))
        self.assertEqual(len(response.history), 3)
        self.test_llm._assemble_prompt.assert_called_once_with(
            message, valid_request.history, persona.model_dump())
        self.test_llm._call_model.assert_called_once_with(
            self.test_llm._assemble_prompt.return_value, valid_request)

        # Request for a different model will raise an exception
        invalid_request = LLMRequest(query=message, history=history,
                                     persona=persona, model="invalid_model")
        with self.assertRaises(ValueError):
            self.test_llm.query_model(invalid_request)

    def test_convert_role(self):
        self.assertEqual(self.test_llm.convert_role("user"), "user")
        self.assertEqual(self.test_llm.convert_role("llm"), "assistant")
        with self.assertRaises(ValueError):
            self.test_llm.convert_role("assistant")
