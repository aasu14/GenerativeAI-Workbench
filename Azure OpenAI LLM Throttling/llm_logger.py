import json
import os
import atexit
import pandas as pd


class LLMLogger:
    logs = None

    def __init__(self, log_file='llm_logger.json'):
        self.log_file = log_file
        if LLMLogger.logs is None:  # Only load logs if they have not been loaded yet
            LLMLogger.logs = []
            self.load_logs()
        atexit.register(self.save_logs)

    def llm_log(self, llm_model_name=None, llm_resource_name=None, embed_model_name=None, embed_resource_name=None,
                func_name=None, token_usage=None, pipeline_stage=None):
        log_entry = {
            'llm_model_name': llm_model_name,
            'llm_resource_name': llm_resource_name,
            'embed_model_name': embed_model_name,
            'embed_resource_name': embed_resource_name,
            'func_name': func_name,
            'token_usage': token_usage,
            'pipeline_stage': pipeline_stage
        }
        # Remove None values from the log entry
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        LLMLogger.logs.append(log_entry)
        self.save_logs()

    def get_logs(self):
        log_df = pd.json_normalize(LLMLogger.logs)
        log_df['pipeline_stage'] = log_df['pipeline_stage'].fillna(method='bfill')
        log_df = log_df.dropna(subset=['llm_model_name'])
        return log_df

    def get_logs_by_pipeline_stage(self, pipeline_stage):
        pipeline_distribution = self.get_logs().groupby(['pipeline_stage', 'func_name']).size().reset_index(
            name='request_count')
        return pipeline_distribution[pipeline_distribution['pipeline_stage'] == pipeline_stage]

    def get_logs_by_resource_name(self):
        return self.get_logs().groupby(
            ['pipeline_stage', 'llm_resource_name', 'embed_resource_name']).size().reset_index(name='request_count')

    def save_logs(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump(LLMLogger.logs, f, indent=4)
        except Exception as e:
            print(f"Failed to save logs: {e}")

    def load_logs(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    LLMLogger.logs = json.load(f)
            except Exception as e:
                print(f"Failed to load logs: {e}")

    def __repr__(self):
        return f"LLMLogger({LLMLogger.logs})"


if __name__ == "__main__":
    logger = LLMLogger()
    try:
        logger.llm_log(llm_model_name='gpt-4o', llm_resource_name='resource_a', embed_model_name='EUS_reg',
                       embed_resource_name=150, func_name='modifier', token_usage=200, pipeline_stage='main.py')
    except Exception as e:
        print(f"An error occurred: {e}")

    print(logger.get_logs())
    print(logger.get_logs_by_pipeline_stage('parse_guidelines'))
    print(logger.get_logs_by_resource_name('resource_c'))
