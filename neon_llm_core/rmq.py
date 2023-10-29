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

from neon_mq_connector.connector import MQConnector
from neon_mq_connector.utils.rabbit_utils import create_mq_callback
from ovos_utils.log import LOG

from neon_llm_core.config import load_config
from neon_llm_core.llm import NeonLLM


class NeonLLMMQConnector(MQConnector, ABC):
    """
        Module for processing MQ requests to Fast Chat LLM
    """

    opinion_prompt = ""

    def __init__(self):
        self.service_name = f'neon_llm_{self.name}'

        self.ovos_config = load_config()
        mq_config = self.ovos_config.get("MQ", None)
        super().__init__(config=mq_config, service_name=self.service_name)
        self.vhost = "/llm"

        self.register_consumers()
        self._model = None

    def register_consumers(self):
        for idx in range(self.model_config["num_parallel_processes"]):
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
        return self.ovos_config.get(f"LLM_{self.name.upper()}", None)
    
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
    def handle_request(self, body: dict):
        """
            Handles ask requests from MQ to LLM
            :param body: request body (dict)
        """
        message_id = body["message_id"]
        routing_key = body["routing_key"]

        query = body["query"]
        history = body["history"]
        persona = body.get("persona",{})

        try:
            response = self.model.ask(message=query, chat_history=history, persona=persona)
        except ValueError as err:
            LOG.error(f'ValueError={err}')
            response = 'Sorry, but I cannot respond to your message at the moment, please try again later'
        api_response = {
            "message_id": message_id,
            "response": response
        }
        self.send_message(request_data=api_response,
                          queue=routing_key)
        LOG.info(f"Handled ask request for message_id={message_id}")

    @create_mq_callback()
    def handle_score_request(self, body: dict):
        """
            Handles score requests from MQ to LLM
            :param body: request body (dict)
        """
        message_id = body["message_id"]
        routing_key = body["routing_key"]

        query = body["query"]
        responses = body["responses"]
        persona = body.get("persona",{})

        if not responses:
            sorted_answer_indexes = []
        else:
            try:
                sorted_answer_indexes = self.model.get_sorted_answer_indexes(question=query, answers=responses, persona=persona)
            except ValueError as err:
                LOG.error(f'ValueError={err}')
                sorted_answer_indexes = []
        api_response = {
            "message_id": message_id,
            "sorted_answer_indexes": sorted_answer_indexes
        }
        self.send_message(request_data=api_response,
                          queue=routing_key)
        LOG.info(f"Handled score request for message_id={message_id}")

    @create_mq_callback()
    def handle_opinion_request(self, body: dict):
        """
            Handles opinion requests from MQ to LLM
            :param body: request body (dict)
        """
        message_id = body["message_id"]
        routing_key = body["routing_key"]

        query = body["query"]
        options = body["options"]
        persona = body.get("persona",{})
        responses = list(options.values())

        if not responses:
            opinion = "Sorry, but I got no options to choose from."
        else:
            try:
                sorted_answer_indexes = self.model.get_sorted_answer_indexes(question=query, answers=responses, persona=persona)
                best_respondent_nick, best_response = list(options.items())[sorted_answer_indexes[0]]
                opinion = self._ask_model_for_opinion(respondent_nick=best_respondent_nick,
                                                      question=query,
                                                      answer=best_response,
                                                      persona=persona)
            except ValueError as err:
                LOG.error(f'ValueError={err}')
                opinion = "Sorry, but I experienced an issue trying to make up an opinion on this topic"

        api_response = {
            "message_id": message_id,
            "opinion": opinion
        }

        self.send_message(request_data=api_response,
                          queue=routing_key)
        LOG.info(f"Handled ask request for message_id={message_id}")

    def _ask_model_for_opinion(self, respondent_nick: str, question: str, answer: str, persona: dict) -> str:
        prompt = self.compose_opinion_prompt(respondent_nick=respondent_nick,
                                             question=question,
                                             answer=answer)
        opinion = self.model.ask(message=prompt, chat_history=[], persona=persona)
        LOG.info(f'Received LLM opinion={opinion}, prompt={prompt}')
        return opinion

    @staticmethod
    @abstractmethod
    def compose_opinion_prompt(respondent_nick: str, question: str, answer: str) -> str:
        pass
