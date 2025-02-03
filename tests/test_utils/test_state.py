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
from unittest.mock import MagicMock, patch

from neon_llm_core.utils.personas.state import PersonaHandlersState
from .utils.factory import PersonaFactory


class TestPersonaHandlersState(unittest.TestCase):

    def setUp(self):
        self.mock_service_name = "test_service"
        self.mock_config = {
            "llm_bots": {
                self.mock_service_name: [PersonaFactory.create_mock_llm_persona(enabled=True) for _ in range(5)]
            },
            "MQ": {"users": {"neon_llm_submind": {"user": "test", "password": "test"}}}
        }
        self.persona_handlers_state = PersonaHandlersState(
            service_name=self.mock_service_name,
            ovos_config=self.mock_config
        )

    def test_initialization(self):
        self.assertEqual(self.persona_handlers_state.service_name, self.mock_service_name)
        self.assertEqual(self.persona_handlers_state.ovos_config, self.mock_config)
        self.assertFalse(self.persona_handlers_state.default_personas_running)
        self.assertDictEqual(self.persona_handlers_state._created_items, {})

    def test_default_personas(self):
        self.assertEqual(
            self.persona_handlers_state.default_personas,
            self.mock_config["llm_bots"][self.mock_service_name]
        )

    def test_has_connected_personas(self):
        self.assertFalse(self.persona_handlers_state.has_connected_personas())
        self.persona_handlers_state._created_items["persona_1"] = MagicMock()
        self.assertTrue(self.persona_handlers_state.has_connected_personas())

    @patch("neon_llm_core.utils.personas.state.LLMBot")
    def test_init_default_personas(self, mock_bot):
        self.persona_handlers_state.init_default_personas()
        self.assertTrue(self.persona_handlers_state.default_personas_running)

    @patch("neon_llm_core.utils.personas.state.LLMBot")
    @patch("neon_data_models.models.LLMPersona", new_callable=PersonaFactory.create_mock_llm_persona, enabled=True)
    def test_add_persona_handler(self, mock_persona, mock_bot):
        result = self.persona_handlers_state.add_persona_handler(mock_persona)
        self.assertIn(mock_persona.id, self.persona_handlers_state._created_items)
        self.assertEqual(result, mock_bot())

    def test_clean_up_all_personas(self):
        mock_personas = [PersonaFactory.create_mock_llm_persona(enabled=True) for _ in range(5)]
        self.persona_handlers_state._created_items = {mock_persona.id: MagicMock()
                                                      for mock_persona in mock_personas}
        self.persona_handlers_state.clean_up_personas()

        self.assertEqual(len(self.persona_handlers_state._created_items), 0)

    def test_clean_up_personas_keeps_specified_item(self):
        mock_personas = [PersonaFactory.create_mock_llm_persona(enabled=True) for _ in range(5)]
        self.persona_handlers_state._created_items = {mock_persona.id: MagicMock()
                                                      for mock_persona in mock_personas}

        persona_to_keep = mock_personas[0]
        self.persona_handlers_state.clean_up_personas(ignore_items=[persona_to_keep])

        self.assertEqual(len(self.persona_handlers_state._created_items), 1)
        self.assertTrue(persona_to_keep.id in self.persona_handlers_state._created_items)

    def test_remove_persona(self):
        mock_item = MagicMock()
        self.persona_handlers_state._created_items = {"persona_1": mock_item}
        self.persona_handlers_state.remove_persona("persona_1")
        mock_item.stop.assert_called_once()
        self.assertNotIn("persona_1", self.persona_handlers_state._created_items)
