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

from abc import abstractmethod, ABC
from threading import Thread
from typing import Optional

from neon_data_models.models.api import LLMRequest
from neon_mq_connector.connector import MQConnector
from neon_mq_connector.utils.rabbit_utils import create_mq_callback
from neon_utils.logger import LOG

from neon_data_models.models.api.mq import (LLMProposeResponse,
                                            LLMDiscussResponse, LLMVoteResponse, LLMDiscussRequest, LLMVoteRequest)

from neon_llm_core.utils.config import load_config
from neon_llm_core.llm import NeonLLM
from neon_llm_core.utils.constants import LLM_VHOST
from neon_llm_core.utils.personas.provider import PersonasProvider


class NeonLLMMQConnector(MQConnector, ABC):
    """
        Module for processing MQ requests to Fast Chat LLM
    """

    async_consumers_enabled = True

    def __init__(self, config: Optional[dict] = None):
        self.service_name = f'neon_llm_{self.name}'

        self.ovos_config = config or load_config()
        mq_config = self.ovos_config.get("MQ", dict())
        super().__init__(config=mq_config, service_name=self.service_name)
        self.vhost = LLM_VHOST

        self.register_consumers()
        self._model = None
        self._bots = list()
        self._personas_provider = PersonasProvider(service_name=self.name,
                                                   ovos_config=self.ovos_config)

    def register_consumers(self):
        for idx in range(self.model_config.get("num_parallel_processes", 1)):
            self.register_consumer(name=f"neon_llm_{self.name}_ask_{idx}",
                                   vhost=self.vhost,
                                   queue=self.queue_ask,
                                   callback=self.handle_request,
                                   on_error=self.default_error_handler,)
        self.register_consumer(name=f'neon_llm_{self.name}_score',
                               vhost=self.vhost,
                               queue=self.queue_score,
                               callback=self.handle_score_request,
                               on_error=self.default_error_handler,)
        self.register_consumer(name=f'neon_llm_{self.name}_discussion',
                               vhost=self.vhost,
                               queue=self.queue_opinion,
                               callback=self.handle_opinion_request,
                               on_error=self.default_error_handler,)
    
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def model_config(self):
        if f"LLM_{self.name.upper()}" not in self.ovos_config:
            LOG.warning(f"No config for {self.name} found in "
                        f"{list(self.ovos_config.keys())}")
        return self.ovos_config.get(f"LLM_{self.name.upper()}", dict())
    
    @property
    def queue_ask(self):
        return f"{self.name}_input"
    
    @property
    def queue_score(self):
        return f"{self.name}_score_input"
    
    @property
    def queue_opinion(self):
        return f"{self.name}_discussion_input"

    @property
    @abstractmethod
    def model(self) -> NeonLLM:
        pass

    @create_mq_callback()
    def handle_request(self, body: dict) -> Thread:
        """
        Handles ask requests (response to prompt) from MQ to LLM
        :param body: request body (dict)
        """
        # Handle this asynchronously so multiple subminds can be handled
        # concurrently
        t = Thread(target=self._handle_request_async, args=(body,),
                   daemon=True)
        t.start()
        return t

    def _handle_request_async(self, request: dict):
        message_id = request["message_id"]
        routing_key = request["routing_key"]

        try:
            response = self.model.query_model(LLMRequest(**request))
        except ValueError as err:
            LOG.error(f'ValueError={err}')
            response = ('Sorry, but I cannot respond to your message at the '
                        'moment, please try again later')
        api_response = LLMProposeResponse(message_id=message_id,
                                          response=response.response,
                                          routing_key=routing_key)
        LOG.info(f"Sending response: {api_response}")
        self.send_message(request_data=api_response.model_dump(),
                          queue=routing_key)
        LOG.info(f"Handled ask request for message_id={message_id}")

    # TODO: Refactor score and opinion to work async like request
    @create_mq_callback()
    def handle_score_request(self, body: dict):
        """
        Handles score requests (vote) from MQ to LLM
        :param body: request body (dict)
        """
        request = LLMVoteRequest(**body)

        if not request.responses:
            sorted_answer_idx = []
        else:
            try:
                sorted_answer_idx = self.model.get_sorted_answer_indexes(
                    question=request.query, answers=request.responses,
                    persona=request.persona.model_dump())
            except ValueError as err:
                LOG.error(f'ValueError={err}')
                sorted_answer_idx = []

        api_response = LLMVoteResponse(message_id=request.message_id,
                                       routing_key=request.routing_key,
                                       sorted_answer_indexes=sorted_answer_idx)
        self.send_message(request_data=api_response.model_dump(),
                          queue=request.routing_key)
        LOG.info(f"Handled score request for message_id={request.message_id}")

    @create_mq_callback()
    def handle_opinion_request(self, body: dict):
        """
        Handles opinion requests (discuss) from MQ to LLM
        :param body: request body (dict)
        """
        request = LLMDiscussRequest(**body)

        if not request.options:
            opinion = "Sorry, but I got no options to choose from."
        else:
            try:
                sorted_answer_indexes = self.model.get_sorted_answer_indexes(
                    question=request.query,
                    answers=list(request.options.values()),
                    persona=request.persona.model_dump())
                best_respondent_nick, best_response = \
                    list(request.options.items())[sorted_answer_indexes[0]]
                opinion = self._ask_model_for_opinion(
                    respondent_nick=best_respondent_nick,
                    llm_request=LLMRequest(**body), answer=best_response)
            except ValueError as err:
                LOG.error(f'ValueError={err}')
                opinion = ("Sorry, but I experienced an issue trying to form "
                           "an opinion on this topic")

        api_response = LLMDiscussResponse(message_id=request.message_id,
                                          routing_key=request.routing_key,
                                          opinion=opinion)
        self.send_message(request_data=api_response.model_dump(),
                          queue=request.routing_key)
        LOG.info(f"Handled ask request for message_id={request.message_id}")

    def _ask_model_for_opinion(self, llm_request: LLMRequest,
                               respondent_nick: str,
                               answer: str) -> str:
        llm_request.query = self.compose_opinion_prompt(
            respondent_nick=respondent_nick, question=llm_request.query,
            answer=answer)
        opinion = self.model.query_model(llm_request)
        LOG.info(f'Received LLM opinion={opinion}, prompt={llm_request.query}')
        return opinion.response

    @staticmethod
    @abstractmethod
    def compose_opinion_prompt(respondent_nick: str, question: str,
                               answer: str) -> str:
        """
        Format a response into a prompt to evaluate another submind's response
        @param respondent_nick: Name of submind providing a response
        @param question: Prompt being responded to
        @param answer: respondent's response to the question
        """
        pass

    def run(self, run_consumers: bool = True, run_sync: bool = True,
            run_observer: bool = True, **kwargs):
        super().run(run_consumers=run_consumers,
                    run_sync=run_sync,
                    run_observer=run_observer,
                    **kwargs)
        self._personas_provider.start_sync()

    def stop(self):
        super().stop()
        self._personas_provider.stop_sync()
