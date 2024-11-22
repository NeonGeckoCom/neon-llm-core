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

from typing import List, Optional
from chatbot_core.v2 import ChatBot
from neon_data_models.models.api.mq import (LLMProposeRequest,
                                            LLMDiscussRequest, LLMVoteRequest, LLMProposeResponse, LLMDiscussResponse,
                                            LLMVoteResponse)
from neon_mq_connector.utils.client_utils import send_mq_request
from neon_utils.logger import LOG
from neon_data_models.models.api.llm import LLMPersona

from neon_llm_core.utils.config import LLMMQConfig


class LLMBot(ChatBot):

    def __init__(self, *args, **kwargs):
        ChatBot.__init__(self, *args, **kwargs)
        self.bot_type = "submind"
        self.base_llm = kwargs["llm_name"]  # chatgpt, fastchat, etc.
        self.persona = kwargs["persona"]
        self.persona = LLMPersona(**self.persona) if \
            isinstance(self.persona, dict) else self.persona
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
        response = self._get_llm_api_response(shout=shout)
        return response.response if response else "I have nothing to say here..."

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
                                            options=options)
        return opinion.opinion if opinion else "I have nothing to say here..."

    def ask_appraiser(self, options: dict, context: dict = None) -> str:
        """
        Selects one of the responses to a prompt and casts a vote in the conversation.
        :param options: proposed responses (botname: response)
        :param context: message context
        """
        if options:
            # Remove self answer from available options
            options = {k: v for k, v in options.items()
                       if k != self.service_name}
            bots = list(options)
            bot_responses = list(options.values())
            LOG.info(f'bots={bots}, answers={bot_responses}')
            prompt = self.prompt_id_to_shout.pop(context['prompt_id'], '')
            answer_data = self._get_llm_api_choice(prompt=prompt,
                                                   responses=bot_responses)
            LOG.info(f'Received answer_data={answer_data}')
            if answer_data and answer_data.sorted_answer_indexes:
                return bots[answer_data.sorted_answer_indexes[0]]
        return "abstain"

    def _get_llm_api_response(self, shout: str) -> Optional[LLMProposeResponse]:
        """
        Requests LLM API for response on provided shout
        :param shout: provided should string
        :returns response from LLM API
        """
        queue = self.mq_queue_config.ask_response_queue
        LOG.info(f"Sending to {self.mq_queue_config.vhost}/{queue}")
        try:
            request_data = LLMProposeRequest(model=self.base_llm,
                                             persona=self.persona,
                                             query=shout,
                                             history=[],
                                             message_id="")
            resp_data = send_mq_request(vhost=self.mq_queue_config.vhost,
                                        request_data=request_data.model_dump(),
                                        target_queue=queue,
                                        response_queue=f"{queue}.response")
            return LLMProposeResponse(**resp_data)
        except Exception as e:
            LOG.exception(f"Failed to get response on "
                          f"{self.mq_queue_config.vhost}/{queue}: {e}")

    def _get_llm_api_opinion(self, prompt: str,
                             options: dict) -> Optional[LLMDiscussResponse]:
        """
        Requests LLM API for discussion of provided submind responses
        :param prompt: incoming prompt text
        :param options: proposed responses (botname: response)
        :returns response data from LLM API
        """
        queue = self.mq_queue_config.ask_discusser_queue
        try:
            request_data = LLMDiscussRequest(model=self.base_llm,
                                             persona=self.persona,
                                             query=prompt,
                                             options=options,
                                             history=[],
                                             message_id="")
            resp_data = send_mq_request(vhost=self.mq_queue_config.vhost,
                                        request_data=request_data.model_dump(),
                                        target_queue=queue,
                                        response_queue=f"{queue}.response")
            return LLMDiscussResponse(**resp_data)
        except Exception as e:
            LOG.exception(f"Failed to get response on "
                          f"{self.mq_queue_config.vhost}/{queue}: {e}")

    def _get_llm_api_choice(self, prompt: str,
                            responses: List[str]) -> Optional[LLMVoteResponse]:
        """
        Requests LLM API for choice among provided message list
        :param prompt: incoming prompt text
        :param responses: list of answers to select from
        :returns response data from LLM API
        """
        queue = self.mq_queue_config.ask_appraiser_queue

        try:
            request_data = LLMVoteRequest(model=self.base_llm,
                                          persona=self.persona,
                                          query=prompt,
                                          responses=responses,
                                          history=[],
                                          message_id="")
            resp_data = send_mq_request(vhost=self.mq_queue_config.vhost,
                                        request_data=request_data.model_dump(),
                                        target_queue=queue,
                                        response_queue=f"{queue}.response")
            return LLMVoteResponse(**resp_data)
        except Exception as e:
            LOG.exception(f"Failed to get response on "
                          f"{self.mq_queue_config.vhost}/{queue}: {e}")

    @staticmethod
    def get_llm_mq_config(llm_name: str) -> LLMMQConfig:
        """
        Get MQ queue names that the LLM service has access to. These are
        LLM-oriented, not bot/persona-oriented.
        """
        return LLMMQConfig(ask_response_queue=f"{llm_name}_input",
                           ask_appraiser_queue=f"{llm_name}_score_input",
                           ask_discusser_queue=f"{llm_name}_discussion_input")
