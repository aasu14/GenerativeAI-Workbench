import logging
import time
from datetime import datetime, timedelta
from azure.mgmt.monitor import MonitorManagementClient, ManagedIdentityCredential
from config import MI_CLIENT_ID, SUBSCRIPTION_ID

# Initialize credentials and monitor client
credential = ManagedIdentityCredential(client_id=MI_CLIENT_ID)
monitor_client = MonitorManagementClient(credential, SUBSCRIPTION_ID)

glob_var = {}

# Define constants
OPENAI_TOKEN_THRESHOLD = 800000
OPENAI_EMBED_TOKEN_THRESHOLD = 100000
OPENAI_REQUEST_THRESHOLD = 5000
OPENAI_EMBED_REQUEST_THRESHOLD = 600


class OpenAIResourceManager:
    def __init__(self):
        self.resource_url = glob_var.RESOURCE_URL

    def get_azure_monitoring_logs(self, service_name, model_name, metric_name):
        self.resource_url = glob_var.RESOURCE_URL
        timespan = f"{(datetime.utcnow() - timedelta(minutes=501)).isoformat()}/{datetime.utcnow().isoformat()}"
        filter_query = f"ModelName eq '{model_name}'" if metric_name != 'Ratelimit' else None
        metrics_data = monitor_client.metrics.list(
            resource_uri=self.resource_url[service_name],
            timespan=timespan,
            interval='PT1M',
            metricnames=metric_name,
            aggregation='Total',
            filter=filter_query
        )
        if metrics_data.value:
            timeseries = metrics_data.value[0].timeseries
            if timeseries:
                usage_1min = [m.total for m in timeseries[0].data][-5:]
                return max(usage_1min) if usage_1min else 0
        return 0

    def get_service_and_model_names(self, client):
        service_name = client.azure_endpoint.split('//')[-1].split('.')[0]
        model_name = client.model_name if hasattr(client, 'model_name') else client.model
        return service_name, model_name

    def get_interval_level_monitoring_logs(self, metric_name, llm_model, embed_model):
        openai_service_name, openai_model_name = self.get_service_and_model_names(llm_model)
        openai_embed_service_name, openai_embed_model_name = self.get_service_and_model_names(embed_model)

        openai_logs = self.get_azure_monitoring_logs(openai_service_name, openai_model_name, metric_name)
        openai_embed_logs = self.get_azure_monitoring_logs(openai_embed_service_name, openai_embed_model_name,
                                                           metric_name)
        return openai_logs, openai_embed_logs

    def get_usage_llm(self, llm_model, embed_model):
        """Helper function to fetch usage statistics for a given region."""
        llm_ratelimit_token, embed_ratelimit_token = self.get_interval_level_monitoring_logs('Ratelimit',
                                                                                             llm_model, embed_model)
        llm_request_count, embed_request_count = (
            self.get_interval_level_monitoring_logs('AzureOpenAIRequests', llm_model, embed_model))
        return [int(llm_ratelimit_token), int(llm_request_count), int(embed_ratelimit_token), int(embed_request_count)]

    def get_llm_resource_throttling(self, data):
        llm_rate_resource, llm_rate_value = min(data, key=lambda x: x[1])[0], min(data, key=lambda x: x[1])[1]
        llm_request_resource, llm_request_value = min(data, key=lambda x: x[2])[0], min(data, key=lambda x: x[2])[2]
        if llm_rate_resource != llm_request_resource:
            if llm_rate_value > OPENAI_TOKEN_THRESHOLD:
                time.sleep(2)
                return llm_request_resource
            elif llm_request_value > OPENAI_REQUEST_THRESHOLD:
                time.sleep(2)
                return llm_rate_resource
            else:
                return llm_rate_resource
        else:
            return llm_rate_resource

    def get_embed_resource_throttling(self, data):
        embed_rate_resource, embed_rate_value = min(data, key=lambda x: x[3])[0], min(data, key=lambda x: x[3])[3]
        embed_request_resource, embed_request_value = min(data, key=lambda x: x[4])[0], min(data, key=lambda x: x[4])[4]
        if embed_rate_resource != embed_request_resource:
            if embed_rate_value > OPENAI_EMBED_TOKEN_THRESHOLD:
                time.sleep(2)
                return embed_request_resource
            elif embed_request_value > OPENAI_EMBED_REQUEST_THRESHOLD:
                time.sleep(2)
                return embed_rate_resource
            else:
                return embed_request_resource
        else:
            return embed_request_resource

    def load_balancing_openai_resource(self):
        llm_init_obj = glob_var.OPENAI_INIT.get_client_object()
        openai_init_keys = list(llm_init_obj.keys())
        llm_usage = []
        for i in range(0, len(openai_init_keys), 2):
            region = openai_init_keys[i].split('_')[-2]
            llm_model, embed_model = llm_init_obj[openai_init_keys[i]], llm_init_obj[openai_init_keys[i + 1]]
            llm_usage.append(tuple([region] + self.get_usage_llm(llm_model, embed_model)))

        llm = llm_init_obj[f'llm_{self.get_llm_resource_throttling(llm_usage)}_{openai_init_keys[0].split("_")[-1]}']
        embed = llm_init_obj[f'embed_{self.get_embed_resource_throttling(llm_usage)}_{openai_init_keys[1].split("_")[-1]}']
        return llm, embed


if __name__ == '__main__':
    # openai_init = OpenAIInitializer(environment='DEV', _model_name='o1-mini', _embed_model_name='text-embedding-ada-002')
    resource_manager = OpenAIResourceManager()
    llm, embed = resource_manager.load_balancing_openai_resource()
