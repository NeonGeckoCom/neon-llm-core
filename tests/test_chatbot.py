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

from neon_data_models.models.api import LLMPersona

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

    def test_get_llm_api_response(self):
        # TODO
        pass

    def test_get_llm_api_opinion(self):
        # TODO
        pass

    def test_get_llm_api_choice(self):
        # TODO
        pass
