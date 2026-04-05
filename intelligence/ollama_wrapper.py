import logging
import time

import requests

from intelligence.llm_wrapper import LLMWrapper
from warning_observer import WarningObserver

logger = logging.getLogger('ollama_wrapper')


class OllamaWrapper(LLMWrapper):
    """
    Class for wrapping around Ollama models.

    :param base_url: base URL of the Ollama server
    :param model_name: model served by Ollama
    :param temperature: temperature used for generation
    :param observers: observers notified on retries
    :param timeout: request timeout in seconds
    :param max_attempt: maximum number of request attempts
    """

    _base_url: str
    _model_name: str
    _temperature: float
    _observers: list[WarningObserver]
    _timeout: int
    _max_attempt: int

    def __init__(self, base_url: str, model_name: str = "llama3.2:3b", temperature: float = 0,
                 observers=None, timeout: int = 120, max_attempt: int = 5):
        super().__init__()
        if observers is None:
            observers = []

        self._base_url = base_url.rstrip('/')
        self._model_name = model_name
        self._temperature = temperature
        self._observers = observers
        self._timeout = timeout
        self._max_attempt = max_attempt

        self._verify_connection()

    def _verify_connection(self) -> None:
        """
        Verify connection with the Ollama server.
        """
        response = requests.get(f"{self._base_url}/api/tags", timeout=10)
        response.raise_for_status()

    def _notify_observers(self, attempt_number: int, outcome: str) -> None:
        """
        Notify observers that a retry occurred.

        :param attempt_number: retry attempt number
        :param outcome: retry outcome information
        """
        for observer in self._observers:
            observer.notify_gpt_retry({
                'attempt number': attempt_number,
                'outcome': outcome
            })

    def make_request(self, message: str) -> str:
        """
        Makes a request to Ollama and returns the response.

        :param message: an input to the model
        :return: response from the model
        """
        logger.debug(f"ollama_input=\"{message}\"")

        payload = {
            "model": self._model_name,
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": self._temperature
            }
        }

        sleep_seconds = 2
        last_exception = None

        for attempt in range(1, self._max_attempt + 1):
            try:
                response = requests.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                    timeout=self._timeout
                )
                response.raise_for_status()
                result = response.json()
                llm_response = result.get('response', '')

                tokens_used = result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                self.total_tokens_used += tokens_used

                logger.debug(f"ollama_output=\"{llm_response}\"")
                return llm_response
            except (requests.exceptions.RequestException, ValueError) as exception:
                last_exception = exception
                self._notify_observers(attempt, str(exception))
                if attempt < self._max_attempt:
                    logger.warning(
                        f"Retrying Ollama request: attempt {attempt} ended with: {exception}")
                    time.sleep(sleep_seconds)
                    sleep_seconds = min(sleep_seconds * 2, 30)

        if last_exception is not None:
            raise last_exception
        return ""
