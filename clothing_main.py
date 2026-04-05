from conv_rec_system import ConvRecSystem
from dotenv import load_dotenv
import logging.config
import warnings
import yaml
import os

"""
Runs clothing conversational recommendation system in terminal 
"""

warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config['PATH_TO_DOMAIN_CONFIGS'] = "domain_specific/configs/clothing_configs"
load_dotenv()

provider = config.get('MODEL_PROVIDER', 'openai').lower()
if provider == 'openai':
    llm_provider_credential = os.environ.get('OPENAI_API_KEY')
    if not llm_provider_credential:
        raise ValueError("MODEL_PROVIDER=openai requires OPENAI_API_KEY")
elif provider == 'alpaca':
    llm_provider_credential = os.environ.get('GRADIO_URL')
    if not llm_provider_credential:
        raise ValueError("MODEL_PROVIDER=alpaca requires GRADIO_URL")
elif provider == 'ollama':
    llm_provider_credential = os.environ.get(
        'OLLAMA_BASE_URL', config.get('OLLAMA_BASE_URL', 'http://localhost:11434'))
else:
    raise ValueError(f"Unsupported MODEL_PROVIDER={provider}")

conv_rec_system = ConvRecSystem(
    config, llm_provider_credential)

conv_rec_system.run()
