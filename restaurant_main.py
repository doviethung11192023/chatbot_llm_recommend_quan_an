from domain_specific.classes.restaurants.geocoding.nominatim_wrapper import NominatimWrapper
from domain_specific.classes.restaurants.geocoding.google_v3_wrapper import GoogleV3Wrapper
from domain_specific.classes.restaurants.location_constraint_merger import LocationConstraintMerger
from domain_specific.classes.restaurants.location_status import LocationStatus
from domain_specific.classes.restaurants.location_filter import LocationFilter
from information_retriever.filter.word_in_filter import WordInFilter

from conv_rec_system import ConvRecSystem
from dotenv import load_dotenv
import logging.config
import warnings
import yaml
import os


"""
Runs restaurant conversational recommendation system in terminal 
"""


warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config['PATH_TO_DOMAIN_CONFIGS'] = "domain_specific/configs/restaurant_configs"

with open(f"{config['PATH_TO_DOMAIN_CONFIGS']}/domain_specific_config.yaml") as f:
    domain_specific_config = yaml.load(f, Loader=yaml.FullLoader)

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

if 'GOOGLE_API_KEY' not in os.environ:
    geocoder = NominatimWrapper(location_bias=domain_specific_config.get("LOCATION_BIAS"))
    
    if geocoder.geocode("edmonton") is None:
        geocoder = None
else:
    geocoder = GoogleV3Wrapper()

if geocoder is None:
    user_filter_objects = [WordInFilter(["location"], "address")]

    conv_rec_system = ConvRecSystem(
        config, llm_provider_credential,
        user_defined_filter=user_filter_objects)
else:
    user_constraint_merger_objects = [LocationConstraintMerger(geocoder)]
    user_constraint_status_objects = [LocationStatus(geocoder)]
    user_filter_objects = [LocationFilter("location", ["latitude", "longitude"], 3, geocoder)]

    conv_rec_system = ConvRecSystem(
        config, llm_provider_credential, user_defined_constraint_mergers=user_constraint_merger_objects,
        user_constraint_status_objects=user_constraint_status_objects,
        user_defined_filter=user_filter_objects)

conv_rec_system.run()
