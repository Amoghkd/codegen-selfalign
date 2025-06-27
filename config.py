# config.py
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

load_dotenv()

# Load API Key
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    raise EnvironmentError("Please set the OPENROUTER_API_KEY environment variable in your .env file.")

# Define models to be used by the agents
# Using free models from OpenRouter for accessibility
FREE_MODELS = {
    "fast": "meta-llama/llama-3.3-8b-instruct:free",
    "reasoning": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    "general": "meta-llama/llama-3.3-8b-instruct:free",
    "coding": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
    "chat": "meta-llama/llama-3.3-8b-instruct:free",
    "compact": "meta-llama/llama-3.3-8b-instruct:free"
}

# Create model info for non-OpenAI models
def create_model_info(model_name):
    """Create ModelInfo for OpenRouter models."""
    return ModelInfo(
        vision=False,
        function_calling=False,
        json_output=True,
        family="unknown",
        structured_output=True
    )

# Function to create a model client for a specific model
def make_llm_config(model_name):
    """Creates a model client for AutoGen v6.0 using OpenRouter (OpenAI-compatible API)."""
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        model_info=create_model_info(model_name),
        timeout=120,
        temperature=0.7,
        max_tokens=4096
    )