import logging
import json
import requests
import openai
from openai import AzureOpenAI
import tiktoken
import time
import re

from config.openai_config import PARAMETERS, OPEN_AI_MODELS_PARAMS


class GenOpenai(object):
    def __init__(self, model_name: str):
        self._openai_version = OPEN_AI_MODELS_PARAMS[model_name]['openai_version']
        self._openai_base = OPEN_AI_MODELS_PARAMS[model_name]['openai_base']
        self._openai_key = OPEN_AI_MODELS_PARAMS[model_name]['opeanai_key']
        self._deployment_name = OPEN_AI_MODELS_PARAMS[model_name]['deployment_name']
        self.azure_openai = self.set_conn_params()

    def set_conn_params(self):
        client = AzureOpenAI(api_key=self._openai_key, api_version=self._openai_version,
                             azure_endpoint=self._openai_base)
        return client

    def get_api_message(self=None, system_message="You are an AI assistant that helps people find information.",
                        user_message=None):
        return [{"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message}]

    def get_qna_api_message(self, curr_qna_msg, response, next_prompt, max_qna_chat_limit):
        resp = 'your response not received'
        if response is not None:
            resp = response['choices'][0]['message']['content']
        next_qna_msg = self.trim_messages_with_msg_limit(curr_qna_msg, max_qna_chat_limit)
        next_qna_msg.extend([{"role": "assistant", "content": resp}, {"role": "user", "content": next_prompt}])
        return next_qna_msg

    def list_deployments(self):
        return openai.Deployment.list()

    def get_deployment(self, deployment_name):
        return openai.Deployment.retrieve(deployment_name)

    def list_global_models(self):
        return openai.Model.list()

    def list_models(self):
        """
        To be modified; requests library not required
        """
        url = openai.api_base + f"/openai/deployments?api-version={self._openai_version}"
        response = requests.get(url, headers={"api-key": self._openai_key})
        data = json.loads(response.content)
        models_list = [item.get("model") for item in data.get("data", []) if "model" in item]
        return models_list

    @staticmethod
    def num_tokens_from_messages(messages, model):
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def trim_messages_to_token_limit(self, messages, overall_max_tokens, max_response_tokens):
        prompt_max_tokens = overall_max_tokens - max_response_tokens
        token_count = self.num_tokens_from_messages(messages, self._deployment_name)
        while token_count > prompt_max_tokens and len(messages) > 3:
            messages.pop(3), messages.pop(3)
            token_count = self.num_tokens_from_messages(messages, self._deployment_name)
        return messages

    def trim_messages_with_msg_limit(self, messages, max_message_limit):
        while len(messages) > max_message_limit:
            messages.pop(3), messages.pop(3)
        return messages

    def show_conversation(self, messages):
        for message in messages:
            logging.info(f"[{message['role'].upper()}] : {message['content']}")

    def fetch_response(self, msg_prompt, parameters):
        try:
            response = self.azure_openai.chat.completions.create(
                model=self._deployment_name,
                messages=msg_prompt,
                temperature=parameters['temperature'],
                max_tokens=parameters['max_tokens'],
                top_p=parameters['top_p'],
                frequency_penalty=parameters['frequency_penalty'],
                presence_penalty=parameters['presence_penalty'],
            )
            return json.loads(response.to_json())

        except KeyError as e:
            logging.warning(e)

        except Exception as e:
            logging.warning(e)
            if e.status_code == 429:
                logging.warning(f"OpenAI API request exceeded rate limit: {e}")
                seconds = re.search(r"retry after (\d+) second", str(e)).group(1)
                logging.info(f'Sleeping for {seconds} seconds')
                time.sleep(int(seconds) + 0.1)
            return None


if __name__ == '__main__':
    model_name = 'gpt_35_turbo'

    genai = GenOpenai(model_name=model_name)
    messages = genai.get_api_message(user_message="Hello, how can you assist me?")
    response = genai.fetch_response(messages, PARAMETERS)
    print(response['choices'][0]['message']['content'])
