# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 NeonGecko.com Inc.
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
import os
from time import time

from neon_mq_connector.utils import RepeatingTimer
from neon_mq_connector.utils.client_utils import send_mq_request
from ovos_utils.log import LOG

from neon_llm_core.utils.constants import LLM_VHOST
from neon_llm_core.utils.personas.models import PersonaModel
from neon_llm_core.utils.personas.state import PersonaHandlersState


class PersonasProvider:

    PERSONA_STATE_TTL = int(os.getenv("PERSONA_STATE_TTL", 15 * 60))
    PERSONA_SYNC_INTERVAL = int(os.getenv("PERSONA_SYNC_INTERVAL", 5 * 60))
    GET_CONFIGURED_PERSONAS_QUEUE = "get_configured_personas"

    def __init__(self, service_name: str, ovos_config: dict):
        self.service_name = service_name
        self._persona_handlers_state = PersonaHandlersState(service_name=service_name,
                                                            ovos_config=ovos_config)
        self._personas = []  # list of personas available for given service
        self._persona_last_sync = 0
        self._persona_sync_thread = None

    @property
    def persona_sync_thread(self):
        """Creates new synchronization thread which fetches Klat personas"""
        if not (isinstance(self._persona_sync_thread, RepeatingTimer) and
                self._persona_sync_thread.is_alive()):
            self._persona_sync_thread = RepeatingTimer(self.PERSONA_SYNC_INTERVAL,
                                                       self._fetch_persona_config)
            self._persona_sync_thread.daemon = True
        return self._persona_sync_thread

    @property
    def personas(self):
        return self._personas

    @personas.setter
    def personas(self, data):
        now = int(time())
        LOG.debug(f'Setting personas={data}')
        if data and isinstance(data, list):
            self._personas = data
            self._persona_last_sync = now
            self._persona_handlers_state.clean_up_personas(ignore_items=self._personas)
        elif now - self._persona_last_sync > self.PERSONA_STATE_TTL:
            LOG.warning(f'Persona state TTL expired, resetting personas config')
            self._personas = []
            self._persona_handlers_state.init_default_handlers()

    def _fetch_persona_config(self):
        response = send_mq_request(vhost=LLM_VHOST,
                                   request_data={"service_name": self.service_name},
                                   target_queue=PersonasProvider.GET_CONFIGURED_PERSONAS_QUEUE)
        response_data = response.get('items', [])
        personas = []
        for item in response_data:
            item.setdefault('name', item.pop('persona_name', None))
            persona = PersonaModel.parse_obj(obj=item)
            self._persona_handlers_state.add_persona_handler(persona=persona)
            personas.append(persona)
        self.personas = personas

    def start_sync(self):
        self._fetch_persona_config()
        self.persona_sync_thread.start()

    def stop_sync(self):
        if self._persona_sync_thread:
            self._persona_sync_thread.cancel()
            self._persona_sync_thread = None
