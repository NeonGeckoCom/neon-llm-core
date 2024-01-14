import time
from typing import Dict, Union

from ovos_utils import LOG

from neon_llm_core.chatbot import LLMBot


class PersonaHandlersState:

    def __init__(self, service_name: str, ovos_config: dict):
        self._created_items: Dict[str, LLMBot] = {}
        self.service_name = service_name
        self.ovos_config = ovos_config
        self.mq_config = ovos_config.get('MQ', {})
        self.init_default_handlers()

    def init_default_handlers(self):
        if self.ovos_config.get("llm_bots", {}).get(self.service_name):
            from neon_llm_core.chatbot import LLMBot
            LOG.info(f"Chatbot(s) configured for: {self.service_name}")
            for persona in self.ovos_config['llm_bots'][self.service_name]:
                self.add_persona_handler(persona=persona)

    def add_persona_handler(self, persona: dict) -> Union[LLMBot, None]:
        persona_name = persona['name']
        if persona_name in list(self._created_items):
            if self._created_items[persona_name].persona != persona:
                LOG.warning(f"Overriding already existing persona: '{persona_name}' with new data={persona}")
                self._created_items[persona_name].stop()
                # time to gracefully stop the submind
                time.sleep(0.5)
            else:
                LOG.warning('Persona config provided is identical to existing, skipping')
                return self._created_items[persona_name]
        if not persona.get('enabled', True):
            LOG.warning(f"Persona disabled: {persona['name']}")
            return
        # Get a configured username to use for LLM submind connections
        if self.mq_config.get("users", {}).get("neon_llm_submind"):
            self.ovos_config["MQ"]["users"][persona['name']] = self.mq_config['users']['neon_llm_submind']
        bot = LLMBot(llm_name=self.service_name, service_name=persona['name'],
                     persona=persona, config=self.ovos_config,
                     vhost="/chatbots")
        bot.run()
        LOG.info(f"Started chatbot: {bot.service_name}")
        self._created_items[persona_name] = bot
        return bot
