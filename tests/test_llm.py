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
        self.get_sorted_answer_indexes = Mock(side_effect=lambda question, answers, persona: [i for i in range(len(answers))])
        self._call_model = Mock(return_value="mock model response")
        self._model = Mock()

    @property
    def tokenizer(self):
        return None

    @property
    def tokenizer_model_name(self) -> str:
        return "mock_tokenizer"

    @property
    def model(self):
        return self._model

    @property
    def llm_model_name(self) -> str:
        return "mock_model"

    @property
    def _system_prompt(self) -> str:
        return "mock system prompt"

    
class TestNeonLLM(TestCase):
    MockLLM.__abstractmethods__ = set()
    config = {"test_config": True}
    
    def setUp(self):
        # Create a new instance for each test to avoid state leaking between tests
        self.test_llm = MockLLM(self.config)
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

    def test_ask_proposer(self):
        """Test the ask_proposer method handles requests correctly"""
        from neon_data_models.models.api import LLMPersona, LLMProposeRequest, LLMProposeResponse
        
        message = "Test proposal"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")
        
        request = LLMProposeRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key"
        )
        
        self.test_llm.query_model = Mock(return_value=LLMResponse(
            response="mock response", 
            history=history + [("llm", "mock response")]
        ))
        
        response = self.test_llm.ask_proposer(request)
        
        self.assertIsInstance(response, LLMProposeResponse)
        self.assertEqual(response.message_id, "test_message_id")
        self.assertEqual(response.routing_key, "test_routing_key")
        self.assertEqual(response.response, "mock response")
        self.test_llm.query_model.assert_called_once_with(request)

    def test_ask_discusser(self):
        """Test the ask_discusser method with various scenarios"""
        from neon_data_models.models.api import LLMPersona, LLMDiscussRequest, LLMDiscussResponse
        
        message = "Test discussion"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")
        
        empty_request = LLMDiscussRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key",
            options={}
        )
        
        response = self.test_llm.ask_discusser(empty_request)
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(response.message_id, "test_message_id")
        self.assertEqual(response.routing_key, "test_routing_key")
        self.assertIsInstance(response.opinion, str)
        # self.assertNotEqual(response.opinion,
        #                     self.test_llm._ask_model_for_opinion.return_value)
        
        options = {"user1": "First option", "user2": "Second option"}
        valid_request = LLMDiscussRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key",
            options=options
        )
        
        self.test_llm._ask_model_for_opinion = Mock(return_value="mock opinion")
        
        response = self.test_llm.ask_discusser(valid_request)
        self.assertIsInstance(response, LLMDiscussResponse)
        self.assertEqual(response.message_id, "test_message_id")
        self.assertEqual(response.routing_key, "test_routing_key")
        self.assertEqual(response.opinion, "mock opinion")
        self.test_llm._ask_model_for_opinion.assert_called_once()
        
        custom_prompt_method = Mock(return_value="Custom prompt")
        self.test_llm._ask_model_for_opinion.reset_mock()
        
        response = self.test_llm.ask_discusser(valid_request, custom_prompt_method)
        self.assertEqual(response.opinion, "mock opinion")
        self.test_llm._ask_model_for_opinion.assert_called_once_with(
            respondent_nick="user1", 
            llm_request=valid_request, 
            answer="First option",
            compose_opinion_prompt=custom_prompt_method
        )
        
        self.test_llm.get_sorted_answer_indexes.side_effect = ValueError("Test error")
        response = self.test_llm.ask_discusser(valid_request)
        self.assertEqual(response.opinion, "Sorry, but I experienced an issue trying to form an opinion on this topic")
        self.test_llm.get_sorted_answer_indexes.side_effect = None

    def test_ask_appraiser(self):
        """Test the ask_appraiser method with various scenarios"""
        from neon_data_models.models.api import LLMPersona, LLMVoteRequest, LLMVoteResponse
        
        message = "Test voting"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")
        
        empty_request = LLMVoteRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key",
            responses=[]
        )
        
        response = self.test_llm.ask_appraiser(empty_request)
        self.assertIsInstance(response, LLMVoteResponse)
        self.assertEqual(response.message_id, "test_message_id")
        self.assertEqual(response.routing_key, "test_routing_key")
        self.assertEqual(response.sorted_answer_indexes, [])
        
        valid_request = LLMVoteRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key",
            responses=["Response 1", "Response 2", "Response 3"]
        )
        
        # self.test_llm.get_sorted_answer_indexes.return_value = [2, 0, 1]
        response = self.test_llm.ask_appraiser(valid_request)
        self.assertIsInstance(response, LLMVoteResponse)

        self.test_llm.get_sorted_answer_indexes.assert_called_once_with(
            question=message,
            answers=["Response 1", "Response 2", "Response 3"],
            persona=persona.model_dump()
        )
        self.assertEqual(response.sorted_answer_indexes, 
                    self.test_llm.get_sorted_answer_indexes("", [1,2,3], {}))
        
        # self.test_llm.get_sorted_answer_indexes = Mock(side_effect=ValueError("Test error"))
        # response = self.test_llm.ask_appraiser(valid_request)
        # self.assertEqual(response.sorted_answer_indexes, [])

    def test_ask_model_for_opinion(self):
        """Test the _ask_model_for_opinion method"""
        from neon_data_models.models.api import LLMPersona, LLMDiscussRequest, LLMResponse
        
        message = "Test opinion"
        history = [["user", "hello"], ["llm", "Hello. How can I help?"]]
        persona = LLMPersona(name="test_persona", description="test persona")
        request = LLMDiscussRequest(
            query=message, 
            history=history,
            persona=persona,
            model=self.test_llm.llm_model_name,
            message_id="test_message_id",
            routing_key="test_routing_key",
            options={"user1": "Option 1"}
        )
        
        compose_prompt = Mock(return_value="Composed prompt")
        
        # self.test_llm.model = Mock()
        self.test_llm.model.query_model = Mock(return_value=LLMResponse(
            response="Generated opinion",
            history=history + [("llm", "Generated opinion")]
        ))
        
        opinion = self.test_llm._ask_model_for_opinion(
            llm_request=request,
            respondent_nick="user1",
            answer="Option 1",
            compose_opinion_prompt=compose_prompt
        )
        
        compose_prompt.assert_called_once_with(
            respondent_nick="user1",
            question=message,
            answer="Option 1"
        )
        self.test_llm.model.query_model.assert_called_once()
        self.assertEqual(opinion, "Generated opinion")
