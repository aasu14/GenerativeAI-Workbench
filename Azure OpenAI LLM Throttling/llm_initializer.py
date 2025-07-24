import logging
import os
import httpx

from llama_index.llms.azure_openai import AzureOpenAI
from openai import AzureOpenAI as OpenAIAzure
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings


def get_resource_url_config():
    return {'subscriptions_key': "subscriptions_value"}


class OpenAIInitializer:
    def __init__(self, environment, _model_name, _embed_model_name):
        self.environment = environment
        self._model_name = _model_name
        self._embed_model_name = _embed_model_name
        self.config = self.load_config()
        self.initialize_resource_names = []
        self.initialize_clients()

    def load_config(self):

        def common_config():
            verify_ssl = os.environ.get("AZURE_OPENAI_VERIFY_SSL", "True").lower() in ("true", "1")
            return {"verify_ssl": verify_ssl}

        config = {
            "DEV": {
                "EastUS":{
                    "gpt-4o" :{
                        **common_config(),
                        "api_version": "2024-08-01-preview",
                        "openai_deployment": "gpt-4o",
                        "openai_model": "gpt-4o",
                        "api_key": "",
                        "endpoint": ""},
                    "text-embedding-ada-002": {
                        **common_config(),
                        "api_version": "2024-08-01-preview",
                        "openai_embedding_deployment": "text-embedding-ada-002",
                        "openai_embedding_model": "text-embedding-ada-002",
                        "api_key": "",
                        "endpoint": ""},
                    },
                "WestUS":{
                    "gpt-4o" :{
                        **common_config(),
                        "api_version": "2024-08-01-preview",
                        "openai_deployment": "gpt-4o",
                        "openai_model": "gpt-4o",
                        "api_key": "",
                        "endpoint": ""},
                    },
            "PROD": {
                "SouthCentralUS":{
                    "gpt-4o" :{
                        **common_config(),
                        "api_version": "2025-01-01-preview",
                        "openai_deployment": "gpt-4o",
                        "openai_model": "gpt-4o",
                        "api_key": "",
                        "endpoint": ""},
                    "text-embedding-ada-002" :{
                        **common_config(),
                        "api_version": "2025-01-01-preview",
                        "openai_embedding_deployment": "text-embedding-ada-002",
                        "openai_embedding_model": "text-embedding-ada-002",
                        "api_key": "",
                        "endpoint": ""},
                    },
            }
        }
        }

        if self.environment not in config:
            raise ValueError("Invalid environment specified")
        return config[self.environment]

    def initialize_clients(self):
        for resource, params in self.config.items():
            if params.get(self._model_name) and params.get(self._embed_model_name):
                llm_name = f'llm_{resource.lower()}_{self._model_name}'
                embed_name = f'embed_{resource.lower()}_{self._embed_model_name}'
                setattr(self, llm_name, self.get_llm_clients(params[self._model_name]))
                setattr(self, embed_name, self.get_embedding_clients(params[self._embed_model_name]))
                self.initialize_resource_names.extend([llm_name, embed_name])

    @staticmethod
    def get_llm_clients(openai_args):
        client = AzureOpenAI(
            model=openai_args['openai_model'],
            deployment_name=openai_args['openai_deployment'],
            temperature=0.0,
            max_retries=5,
            timeout=300,
            api_key=openai_args['api_key'],
            azure_endpoint=openai_args['endpoint'],
            api_version=openai_args['api_version'],
            http_client=httpx.Client(verify=openai_args['verify_ssl'])
        )
        return client

    @staticmethod
    def get_embedding_clients(openai_args):
        client_embedding = AzureOpenAIEmbedding(
            model=openai_args['openai_embedding_model'],
            deployment_name=openai_args['openai_embedding_deployment'],
            api_key=openai_args['api_key'],
            azure_endpoint=openai_args['endpoint'],
            api_version=openai_args['api_version'],
            http_client=httpx.Client(verify=openai_args['verify_ssl'])
        )
        return client_embedding

    def get_client_names(self):
        return self.initialize_resource_names

    def get_client_object(self):
        return {name: getattr(self, name) for name in self.initialize_resource_names}


if __name__ == '__main__':
    openai_init = OpenAIInitializer(environment='DEV', _model_name='gpt-4o', _embed_model_name='text-embedding-ada-002')
    print(openai_init.get_client_names())
    print(openai_init.get_client_object().keys())
