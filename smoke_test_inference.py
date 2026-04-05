import argparse
import os

import yaml
from dotenv import load_dotenv

from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper
from intelligence.gpt_wrapper import GPTWrapper
from intelligence.ollama_wrapper import OllamaWrapper


def _load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def _build_wrapper(provider: str, model: str, config: dict):
    provider = provider.lower()

    if provider == 'openai':
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError('OPENAI_API_KEY is required when provider=openai')
        return GPTWrapper(openai_key, model_name=model)

    if provider == 'alpaca':
        gradio_url = os.environ.get('GRADIO_URL')
        if not gradio_url:
            raise ValueError('GRADIO_URL is required when provider=alpaca')
        return AlpacaLoraWrapper(gradio_url)

    if provider == 'ollama':
        base_url = os.environ.get(
            'OLLAMA_BASE_URL',
            config.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        )
        return OllamaWrapper(base_url, model_name=model)

    raise ValueError(f'Unsupported provider: {provider}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Quick one-turn inference smoke test for llm-convrec providers.'
    )
    parser.add_argument('--config', default='system_config.yaml',
                        help='Path to system config yaml.')
    parser.add_argument('--provider', default=None,
                        help='Override provider (openai/alpaca/ollama).')
    parser.add_argument('--model', default=None,
                        help='Override model name.')
    parser.add_argument('--prompt', default='Hello! Please introduce yourself in one short sentence.',
                        help='Prompt used for smoke test.')

    args = parser.parse_args()

    load_dotenv()
    config = _load_config(args.config)

    provider = args.provider or config.get('MODEL_PROVIDER', 'openai')
    model = args.model or config.get('MODEL', 'gpt-3.5-turbo')

    print(f'[smoke-test] provider={provider}, model={model}')
    wrapper = _build_wrapper(provider, model, config)

    response = wrapper.make_request(args.prompt)

    if not response or not response.strip():
        raise RuntimeError('Smoke test failed: empty model response.')

    print('[smoke-test] SUCCESS')
    print(f'[smoke-test] prompt: {args.prompt}')
    print(f'[smoke-test] response: {response.strip()}')
    print(f'[smoke-test] tokens_used_so_far: {wrapper.get_total_tokens_used()}')


if __name__ == '__main__':
    main()
