# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
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

from typing import List
from chatbot_core.v2 import ChatBot
from neon_mq_connector.utils.client_utils import send_mq_request
from ovos_utils.log import LOG

from neon_llm_core.utils.config import LLMMQConfig


class LLMBot(ChatBot):

    def __init__(self, *args, **kwargs):
        ChatBot.__init__(self, *args, **kwargs)
        self.bot_type = "submind"
        self.base_llm = kwargs.get("llm_name")  # chatgpt, fastchat, etc.
        self.persona = kwargs.get("persona")
        self.mq_queue_config = self.get_llm_mq_config(self.base_llm)
        LOG.info(f'Initialised config for llm={self.base_llm}|'
                 f'persona={self._bot_id}')
        self.prompt_id_to_shout = dict()

    @property
    def contextual_api_supported(self):
        return True

    def ask_chatbot(self, user: str, shout: str, timestamp: str,
                    context: dict = None) -> str:
        """
        Handles an incoming shout into the current conversation
        :param user: user associated with shout
        :param shout: text shouted by user
        :param timestamp: formatted timestamp of shout
        :param context: message context
        """
        prompt_id = context.get('prompt_id')
        if prompt_id:
            self.prompt_id_to_shout[prompt_id] = shout
        LOG.debug(f"Getting response to {shout}")
        response = self._get_llm_api_response(
            shout=shout).get("response", "I have nothing to say here...")
        return response

    def ask_discusser(self, options: dict, context: dict = None) -> str:
        """
        Provides one discussion response based on the given options

        :param options: proposed responses (botname: response)
        :param context: message context
        """
        options = {k: v for k, v in options.items() if k != self.service_name}
        prompt_sentence = self.prompt_id_to_shout.get(context['prompt_id'], '')
        LOG.info(f'prompt_sentence={prompt_sentence}, options={options}')
        opinion = self._get_llm_api_opinion(prompt=prompt_sentence,
                                            options=options).get('opinion', '')
        return opinion

    def ask_appraiser(self, options: dict, context: dict = None) -> str:
        """
        Selects one of the responses to a prompt and casts a vote in the conversation.
        :param options: proposed responses (botname: response)
        :param context: message context
        """
        if options:
            options = {k: v for k, v in options.items()
                       if k != self.service_name}
            bots = list(options)
            bot_responses = list(options.values())
            LOG.info(f'bots={bots}, answers={bot_responses}')
            prompt = self.prompt_id_to_shout.pop(context['prompt_id'], '')
            answer_data = self._get_llm_api_choice(prompt=prompt,
                                                   responses=bot_responses)
            LOG.info(f'Received answer_data={answer_data}')
            sorted_answer_indexes = answer_data.get('sorted_answer_indexes')
            if sorted_answer_indexes:
                return bots[sorted_answer_indexes[0]]
        return "abstain"

    def _get_llm_api_response(self, shout: str) -> dict:
        """
        Requests LLM API for response on provided shout
        :param shout: provided should string
        :returns response string from LLM API
        """
        queue = self.mq_queue_config.ask_response_queue
        LOG.info(f"Sending to {self.mq_queue_config.vhost}/{queue}")
        try:
            return send_mq_request(vhost=self.mq_queue_config.vhost,
                                   request_data={"query": shout,
                                                 "history": [],
                                                 "persona": self.persona},
                                   target_queue=queue,
                                   response_queue=f"{queue}.response")
        except Exception as e:
            LOG.exception(f"Failed to get response on "
                          f"{self.mq_queue_config.vhost}/"
                          f"{self.mq_queue_config.ask_response_queue}: "
                          f"{e}")
            return dict()

    def _get_llm_api_opinion(self, prompt: str, options: dict) -> dict:
        """
        Requests LLM API for opinion on provided submind responses
        :param prompt: incoming prompt text
        :param options: proposed responses (botname: response)
        :returns response data from LLM API
        """
        queue = self.mq_queue_config.ask_discusser_queue
        return send_mq_request(vhost=self.mq_queue_config.vhost,
                               request_data={"query": prompt,
                                             "options": options,
                                             "persona": self.persona},
                               target_queue=queue,
                               response_queue=f"{queue}.response")

    def _get_llm_api_choice(self, prompt: str, responses: List[str]) -> dict:
        """
        Requests LLM API for choice among provided message list
        :param prompt: incoming prompt text
        :param responses: list of answers to select from
        :returns response data from LLM API
        """
        queue = self.mq_queue_config.ask_appraiser_queue
        return send_mq_request(vhost=self.mq_queue_config.vhost,
                               request_data={"query": prompt,
                                             "responses": responses,
                                             "persona": self.persona},
                               target_queue=queue,
                               response_queue=f"{queue}.response")

    @staticmethod
    def get_llm_mq_config(llm_name: str) -> LLMMQConfig:
        """
        Get MQ queue names that the LLM service has access to. These are
        LLM-oriented, not bot/persona-oriented.
        """
        return LLMMQConfig(ask_response_queue=f"{llm_name}_input",
                           ask_appraiser_queue=f"{llm_name}_score_input",
                           ask_discusser_queue=f"{llm_name}_discussion_input")
