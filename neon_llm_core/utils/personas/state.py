# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2022 Neongecko.com Inc.
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
from typing import Dict, Union, List

from ovos_utils import LOG

from neon_llm_core.chatbot import LLMBot
from neon_llm_core.utils.personas.models import PersonaModel


class PersonaHandlersState:

    def __init__(self, service_name: str, ovos_config: dict):
        self._created_items: Dict[str, LLMBot] = {}
        self.service_name = service_name
        self.ovos_config = ovos_config
        self.mq_config = ovos_config.get('MQ', {})
        self.init_default_handlers()

    def init_default_handlers(self):
        self._created_items = {}
        if self.ovos_config.get("llm_bots", {}).get(self.service_name):
            LOG.info(f"Chatbot(s) configured for: {self.service_name}")
            for persona in self.ovos_config['llm_bots'][self.service_name]:
                self.add_persona_handler(persona=PersonaModel.parse_obj(obj=persona))

    def add_persona_handler(self, persona: PersonaModel) -> Union[LLMBot, None]:
        persona_dict = persona.model_dump()
        if persona.id in list(self._created_items):
            if self._created_items[persona.id].persona != persona_dict:
                LOG.warning(f"Overriding already existing persona: '{persona.id}' with new data={persona}")
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

    def clean_up_personas(self, ignore_items: List[PersonaModel] = None):
        connected_personas = set(self._created_items)
        ignored_persona_ids = set(persona.id for persona in ignore_items or [])
        personas_to_remove = connected_personas - ignored_persona_ids
        for persona_id in personas_to_remove:
            LOG.info(f'Removing persona_id = {persona_id}')
            self._created_items.pop(persona_id, None)
