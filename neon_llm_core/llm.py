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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from neon_data_models.models.api import LLMRequest, LLMResponse
from openai import OpenAI
from ovos_utils.log import log_deprecation


class NeonLLM(ABC):

    mq_to_llm_role = {}

    def __init__(self, config: dict):
        """
        @param config: Dict LLM configuration for this specific LLM
        """
        self._llm_config = config

    @property
    def llm_config(self):
        """
        Get the configuration for this LLM instance
        """
        return self._llm_config

    @property
    @abstractmethod
    def tokenizer(self) -> Optional[object]:
        """
        Get a Tokenizer object for the loaded model, if available.
        :return: optional transformers.PreTrainedTokenizerBase object
        """
        pass

    @property
    @abstractmethod
    def tokenizer_model_name(self) -> str:
        """
        Get a string tokenizer model name (i.e. a Huggingface `model id`)
        associated with `self.tokenizer`.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> OpenAI:
        """
        Get an OpenAI client object to send requests to.
        """
        pass

    @property
    @abstractmethod
    def llm_model_name(self) -> str:
        """
        Get a string model name for the configured `model`
        """
        pass

    @property
    @abstractmethod
    def _system_prompt(self) -> str:
        """
        Get a default string system prompt to use when not included in requests
        """
        pass

    def ask(self, message: str, chat_history: List[List[str]], persona: dict) -> str:
        """
        Generates llm response based on user message and (user, llm) chat history
        """
        log_deprecation("This method is replaced by `query_model` which "
                        "accepts a single `LLMRequest` arg", "1.0.0")
        prompt = self._assemble_prompt(message, chat_history, persona)
        llm_text_output = self._call_model(prompt)
        return llm_text_output

    def query_model(self, request: LLMRequest) -> LLMResponse:
        """
        Calls `self._assemble_prompt` to allow subclass to modify the input
        query and then passes the updated query to `self._call_model`
        :param request: LLMRequest object to generate a response to
        :return:
        """
        if request.model != self.llm_model_name:
            raise ValueError(f"Requested model ({request.model}) is not this "
                             f"model ({self.llm_model_name}")
        request.query = self._assemble_prompt(request.query, request.history,
                                              request.persona.model_dump())
        response = self._call_model(request.query, request)
        history = request.history + [("llm", response)]
        return LLMResponse(response=response, history=history)

    @abstractmethod
    def get_sorted_answer_indexes(self, question: str, answers: List[str], persona: dict) -> List[int]:
        """
        Creates sorted list of answer indexes with respect to order provided in
        `answers`. Results should be sorted from best to worst
        :param question: incoming question
        :param answers: list of answers to rank
        :param persona: dict representation of Persona to use for sorting
        :returns list of indexes
        """
        pass

    @abstractmethod
    def _call_model(self, prompt: str,
                    request: Optional[LLMRequest] = None) -> str:
        """
        Wrapper for Model generation logic. This method may be called
        asynchronously, so it is up to the extending class to use locks or
        queue inputs as necessary.
        :param prompt: Input text sequence
        :param request: Optional LLMRequest object containing parameters to
            include in model requests
        :returns: Output text sequence generated by model
        """
        pass

    @abstractmethod
    def _assemble_prompt(self, message: str,
                         chat_history: List[Union[List[str], Tuple[str, str]]],
                         persona: dict) -> str:
        """
        Assemble the prompt to send to the LLM
        :param message: Input prompt to optionally modify
        :param chat_history: History of preceding conversation
        :param persona: dict representation of Persona that is requested
        :returns: assembled prompt string
        """
        pass

    @abstractmethod
    def _tokenize(self, prompt: str) -> List[str]:
        """
        Tokenize the input prompt into a list of strings
        :param prompt: Input to tokenize
        :return: Tokenized representation of input prompt
        """
        pass

    @classmethod
    def convert_role(cls, role: str) -> str:
        """
        Maps MQ role to LLM's internal domain
        :param role: Role in Neon LLM format
        :return: Role in LLM internal format
        """
        matching_llm_role = cls.mq_to_llm_role.get(role)
        if not matching_llm_role:
            raise ValueError(f"role={role} is undefined, supported are: "
                             f"{list(cls.mq_to_llm_role)}")
        return matching_llm_role
