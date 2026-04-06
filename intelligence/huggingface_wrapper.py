import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intelligence.llm_wrapper import LLMWrapper
from warning_observer import WarningObserver

logger = logging.getLogger('huggingface_wrapper')


class HuggingFaceWrapper(LLMWrapper):
    """
    Class for wrapping around a Hugging Face causal language model.

    This wrapper is intended for chat/instruct models such as Qwen3-4B-Instruct.

    :param model_name: Hugging Face model id or local path
    :param observers: observers notified when loading/generation issues occur
    :param hf_token: optional Hugging Face token for private models
    :param max_new_tokens: max number of newly generated tokens
    :param temperature: generation temperature
    :param top_p: nucleus sampling top-p
    """

    _model_name: str
    _hf_token: Optional[str]
    _observers: list[WarningObserver]
    _max_new_tokens: int
    _temperature: float
    _top_p: float

    _tokenizer = None
    _model = None

    def __init__(self, model_name: str = 'Qwen/Qwen3-4B-Instruct', observers=None,
                 hf_token: Optional[str] = None, max_new_tokens: int = 512,
                 temperature: float = 0.0, top_p: float = 0.9):
        super().__init__()
        if observers is None:
            observers = []

        self._model_name = model_name
        self._hf_token = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
        self._observers = observers
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

        self._load_model()

    def _notify_observers(self, message: str) -> None:
        """
        Notify observers that model loading or generation ran into a recoverable issue.
        """
        for observer in self._observers:
            observer.notify_warning(message)

    def _load_model(self) -> None:
        """
        Load tokenizer and model from Hugging Face.
        The download is cached locally by Hugging Face after the first run.
        """
        try:
            logger.info(f'Loading Hugging Face model: {self._model_name}')
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                token=self._hf_token,
                trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                token=self._hf_token,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            self._model.eval()
        except Exception as exception:
            self._notify_observers(f'Failed to load Hugging Face model {self._model_name}: {exception}')
            raise

    def _build_messages(self, message: str) -> list[dict[str, str]]:
        """
        Wrap the raw user message into a simple chat format.
        """
        return [{'role': 'user', 'content': message}]

    def make_request(self, message: str) -> str:
        """
        Generate a response using the loaded Hugging Face model.
        """
        logger.debug(f'hf_input="{message}"')

        messages = self._build_messages(message)

        try:
            if hasattr(self._tokenizer, 'apply_chat_template'):
                prompt_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt_text = message

            inputs = self._tokenizer(
                prompt_text,
                return_tensors='pt'
            )

            device = getattr(self._model, 'device', None)
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}

            do_sample = self._temperature > 0
            generation_kwargs = {
                'max_new_tokens': self._max_new_tokens,
                'do_sample': do_sample,
                'pad_token_id': self._tokenizer.eos_token_id
            }
            if do_sample:
                generation_kwargs['temperature'] = self._temperature
                generation_kwargs['top_p'] = self._top_p

            with torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **generation_kwargs)

            prompt_length = inputs['input_ids'].shape[-1]
            new_tokens = generated_ids[0][prompt_length:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            self.total_tokens_used += int(new_tokens.shape[-1])
            logger.debug(f'hf_output="{response}"')
            return response
        except Exception as exception:
            logger.exception('Hugging Face generation failed')
            self._notify_observers(f'Hugging Face generation failed for {self._model_name}: {exception}')
            raise
