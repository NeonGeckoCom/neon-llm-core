# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
# Contributors: Daniel McKnight, Guy Daniels, Elon Gasper, Richard Leeds,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
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

import time
from functools import cached_property
from threading import Lock
from typing import Dict, List, Optional

from neon_data_models.models.api.llm import LLMPersona
from neon_utils.logger import LOG
from neon_llm_core.chatbot import LLMBot


class PersonaHandlersState:
    """
    This works with the PersonasProvider object to manage LLMBot instances for
    all configured personas.
    """

    def __init__(self, service_name: str, ovos_config: dict):
        self._created_items: Dict[str, LLMBot] = {}
        self.service_name = service_name
        self.ovos_config = ovos_config
        self.mq_config = ovos_config.get('MQ', {})
        self.default_personas_running = False

        self.personas_clean_up_lock = Lock()
        self.personas_remove_lock = Lock()

    @cached_property
    def default_personas(self):
        return self.ovos_config.get("llm_bots", {}).get(self.service_name, [])

    def has_connected_personas(self) -> bool:
        return bool(self._created_items)

    def init_default_personas(self):
        """
        Initializes LLMBot instances for all personas defined in configuration.
        """
        if self.default_personas and not self.default_personas_running:
            self.clean_up_personas()
            LOG.info(f"Initializing default personas for: {self.service_name}")
            for persona in self.default_personas:
                self.add_persona_handler(
                    persona=LLMPersona.model_validate(obj=persona)
                )
            self.default_personas_running = True

    def add_persona_handler(self, persona: LLMPersona) -> Optional[LLMBot]:
        """
        Creates an `LLMBot` instance for the given persona if the persona does
        not yet exist AND the persona is not disabled in configuration.
        :param persona: Persona definition to generate an LLMBot instance of
        :return: New or existing LLMBot instance or None if the persona is
                 disabled
        """
        persona_dict = persona.model_dump()
        if persona.id in list(self._created_items):
            if self._created_items[persona.id].persona != persona_dict:
                LOG.warning(f"Overriding already existing persona: "
                            f"'{persona.id}' with new data={persona}")
                try:
                    self._created_items[persona.id].stop()
                    # time to gracefully stop the submind
                    time.sleep(0.5)
                except Exception as ex:
                    LOG.warning(f'Failed to gracefully stop {persona.id}, ex={str(ex)}')
            else:
                LOG.debug('Persona config provided is identical to existing, skipping')
                return self._created_items[persona.id]
        if not persona.enabled:
            LOG.warning(f"Persona disabled: '{persona.id}'")
            return
        # Get a configured username to use for LLM submind connections
        persona_id = f"{persona.id}_{self.service_name}"
        self.ovos_config["MQ"]["users"][persona_id] = self.mq_config['users']['neon_llm_submind']
        bot = LLMBot(llm_name=self.service_name, service_name=persona_id,
                     persona=persona_dict, config=self.ovos_config,
                     vhost="/chatbots")
        bot.run()
        LOG.info(f"Started chatbot: {bot.service_name}")
        self._created_items[persona.id] = bot
        return bot

    def clean_up_personas(self, ignore_items: List[LLMPersona] = None):
        with self.personas_clean_up_lock:
            connected_personas = set(self._created_items)
            ignored_persona_ids = set(persona.id for persona in ignore_items or [])
            personas_to_remove = connected_personas - ignored_persona_ids
            for persona_id in personas_to_remove:
                self.remove_persona(persona_id=persona_id)

    def remove_persona(self, persona_id: str):
        with self.personas_remove_lock:
            if persona_id in self._created_items:
                LOG.info(f'Removing persona_id = {persona_id}')
                self._created_items[persona_id].stop()
                self._created_items.pop(persona_id, None)

                if not self.has_connected_personas() and self.default_personas_running:
                    LOG.info("All default personas stopped")
                    self.default_personas_running = False
