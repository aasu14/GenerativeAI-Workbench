from collections import Counter
import re
import json
import pandas as pd
import logging

from util.azure_blob_ops import BlobOperator


class CompletionHandler(object):
    def __init__(self):
        self.blob_operator = BlobOperator()

    def save_to_json(self, data, filename):
        try:
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=4, default=str)
            return True
        except Exception as e:
            logging.error(f"Error saving JSON file: {e}")
            return False

    def load_from_json(self, filename):
        try:
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
            return data
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            return None

    def pretty_print(self, data):
        logging.info(json.dumps(data, indent=4))

    def get_csv_file(self, csv_file):
        data = pd.read_csv(csv_file)
        return data

    def json_to_csv(self, json_dict, file):
        pd.DataFrame(json_dict).to_csv(f"{file}", index=False)


def unique_responses(prompts_result):
    responses = [re.sub(r'\s+', ' ', item['Response'].lower()).strip() for item in prompts_result]
    response_counts = Counter(','.join(responses).split(','))
    df = pd.DataFrame(response_counts.items(), columns=['Unique_Response', 'frequency'])
    df['Unique_Response'] = df['Unique_Response'].str.replace('[^a-zA-Z0-9\s$]', '', regex=True).str.strip()
    unique_resp = df.groupby('Unique_Response', as_index=False)['frequency'].sum()
    return unique_resp


def save_output_files(client, min_date, max_date, temperature, top_p, input_notes, prompts_result, unique_actions_df):
    blobName = f'NBA/{client}/gpt/{min_date}_{max_date}/output'
    suffix = f'{min_date}_{max_date}_T{temperature}_P{top_p}'
    note_input_file = f"{client}_input_notes_{suffix}.csv"
    prompt_res_file = f"{client}_openai_output_responses_{suffix}.csv"
    response_file = f"{client}_openai_output_response_json_{suffix}.json"
    unique_actions_file = f"{client}_unique_actions_{suffix}.csv"

    CompletionHandler().save_to_json(prompts_result, response_file)
    unique_actions_df.to_csv(f"{unique_actions_file}", index=False)
    CompletionHandler().json_to_csv(input_notes, note_input_file)
    CompletionHandler().json_to_csv(prompts_result, prompt_res_file)

    for file in [response_file, note_input_file, prompt_res_file, unique_actions_file]:
        logging.info(f'Uploading on Azure: {file}')
        BlobOperator().upload_blob(local_file_path=file, azure_blob_path=f"{blobName}/{file}", overwrite=True)
    logging.info(f'Uploaded output files on Azure: {blobName}')


def save_output_prompt_files(client, min_date, max_date, temperature, top_p, input_prompts, prompts_result):
    blobName = f'NBA/{client}/gpt/{min_date}_{max_date}/categorized_response'
    suffix = f'{min_date}_{max_date}_T{temperature}_P{top_p}_test'
    prompt_input_file = f"{client}_input_prompts_{suffix}.csv"
    prompt_res_file = f"{client}_openai_prompt_category_{suffix}.csv"

    CompletionHandler().json_to_csv(input_prompts, prompt_input_file)
    CompletionHandler().json_to_csv(prompts_result, prompt_res_file)

    for file in [prompt_input_file, prompt_res_file]:
        logging.info(f'Uploading on Azure: {file}')
        BlobOperator().upload_blob(local_file_path=file,azure_blob_path=f"{blobName}/{file}",overwrite=True)
    logging.info(f'Uploaded output files on Azure: {blobName}')
