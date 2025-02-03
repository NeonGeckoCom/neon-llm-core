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

import unittest
from unittest.mock import patch, Mock, MagicMock

from neon_llm_core.utils.personas.provider import PersonasProvider
from neon_llm_core.utils.personas.state import PersonaHandlersState

from .utils.factory import PersonaFactory


class TestPersonasProvider(unittest.TestCase):

    def setUp(self):
        """Set up the test environment and mock dependencies."""
        self.mock_service_name = "mock_service"
        self.mock_config = {
            "llm_bots": {
                self.mock_service_name: [PersonaFactory.create_mock_llm_persona(enabled=True) for _ in range(5)]
            },
            "MQ": {"users": {"neon_llm_submind": {"user": "test", "password": "test"}}}
        }
        self.provider = PersonasProvider(service_name=self.mock_service_name, ovos_config=self.mock_config)

    @patch("neon_llm_core.utils.personas.provider.send_mq_request")
    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test_fetch_persona_config_success(self, mock_bot, mock_send_mq_request):
        mock_persona_data = [
            PersonaFactory.create_mock_llm_persona(enabled=True).model_dump(),
            PersonaFactory.create_mock_llm_persona(enabled=False).model_dump(),
        ]
        mock_send_mq_request.return_value = {"items": mock_persona_data}
        self.provider._fetch_persona_config()

        self.assertEqual(len(self.provider.personas), 1)
        self.assertEqual(self.provider.personas[0].name, mock_persona_data[0]["name"])

    @patch("neon_llm_core.utils.personas.provider.send_mq_request")
    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test_fetch_persona_config_empty(self, mock_bot, mock_send_mq_request):
        mock_send_mq_request.return_value = {"items": []}
        self.provider._fetch_persona_config()

        self.assertEqual(len(self.provider.personas), 0)

    def test__validate_persona_data_success(self):
        persona_data = PersonaFactory.create_mock_llm_persona().model_dump()
        persona = self.provider._validate_persona_data(persona_data)

        self.assertIsNotNone(persona)
        self.assertEqual(persona.name, persona_data['name'])

    def test__validate_persona_data_returns_None_on_failure(self):
        persona_data = {}
        persona = self.provider._validate_persona_data(persona_data)

        self.assertIsNone(persona)

    def test__validate_persona_identity_success(self):
        persona_identity = PersonaFactory.create_mock_persona_identity().model_dump()
        persona = self.provider._validate_persona_identity(persona_identity)

        self.assertIsNotNone(persona)
        self.assertEqual(persona.name, persona_identity['name'])

    def test__validate_persona_identity_returns_None_on_failure(self):
        persona_identity_data = {}
        persona_identity = self.provider._validate_persona_identity(persona_identity_data)

        self.assertIsNone(persona_identity)

    @patch.object(PersonaHandlersState, "add_persona_handler")
    def test__add_persona_success(self, mock_add_persona_handler):
        mock_add_persona_handler.return_value = True
        persona = Mock(id="mock_persona")

        result = self.provider._add_persona(persona)

        self.assertTrue(result)
        mock_add_persona_handler.assert_called_once_with(persona=persona)

    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test__add_persona_when_default_personas_are_running(self, mock_llm_bot):
        persona = PersonaFactory.create_mock_llm_persona(enabled=True)

        self.provider._persona_handlers_state.init_default_personas()
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id for persona in self.mock_config['llm_bots'][self.mock_service_name]])

        result = self.provider._add_persona(persona)
        self.assertTrue(result)

        # Defaults personas eliminated
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id])

    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test__add_disabled_persona_when_default_personas_are_running(self, mock_llm_bot):
        persona = PersonaFactory.create_mock_llm_persona(enabled=False)

        self.provider._persona_handlers_state.init_default_personas()
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id for persona in self.mock_config['llm_bots'][self.mock_service_name]])

        result = self.provider._add_persona(persona)
        self.assertFalse(result)

        # Defaults personas stay as is
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id for persona in self.mock_config['llm_bots'][self.mock_service_name]])

    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test_disable_latest_persona_should_activate_default_personas(self, mock_llm_bot):
        persona = PersonaFactory.create_mock_llm_persona(enabled=True)

        self.provider._add_persona(persona)

        # Persona is connected
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id])

        persona.enabled = False

        self.provider._add_persona(persona)

        # Persona is disconnected and default personas are initialised
        self.assertEqual(self.provider._persona_handlers_state.connected_persona_ids,
                         [persona.id for persona in self.mock_config['llm_bots'][self.mock_service_name]])

    def test_should_reset_personas_when_no_data(self):
        result = self.provider._should_reset_personas([])
        self.assertTrue(result)

    def test_should_not_reset_personas_with_data_and_no_sync_interval(self):
        self.provider.PERSONA_SYNC_INTERVAL = 0
        result = self.provider._should_reset_personas([MagicMock()])
        self.assertFalse(result)
