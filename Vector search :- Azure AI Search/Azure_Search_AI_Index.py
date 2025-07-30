import json
import uuid
from config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_COGNITIVE_SEARCH_SERVICE_NAME, AZURE_COGNITIVE_SEARCH_API_KEY, AZURE_COGNITIVE_SEARCH_INDEX_NAME, azure_openai_embedding_deployment, blob_container_name, blob_connection_string

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents import SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    CorsOptions

)
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from azure.search.documents.indexes.models import (
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField
)
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.search.documents.models import VectorizedQuery
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def add_metadata(documents, directory):
    for doc in documents:
        doc.metadata.update({"department": directory})


index_name = "testingindex"
AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)


def upload_doc_to_index():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_openai_embedding_deployment,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
    )
    embedding_function = embeddings.embed_query
    docs = []

    for doc in texts:
        docs.append({
            "id": str(uuid.uuid4()),
            "content": doc.page_content,
            "content_vector": embedding_function(doc.page_content),
            "metadata": str(doc.metadata)
        })

    json_data = json.dumps(docs)

    with open('test.json', "w") as f:
        f.write(json_data)

    with open('test.json', 'r') as f:
        documents = json.load(f)

    with SearchIndexingBufferedSender(
            endpoint=AZURE_COGNITIVE_SEARCH_SERVICE_NAME,
            index_name=index_name,
            credential=AZURE_COGNITIVE_SEARCH_CREDENTIAL,
    ) as batch_client:
        result_update = batch_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents in total")


def get_index(index_name):
    AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)
    client = SearchIndexClient(AZURE_COGNITIVE_SEARCH_SERVICE_NAME, AZURE_COGNITIVE_SEARCH_CREDENTIAL)
    client.get_index(index_name)


def keyword_search(search_client, query):
    results = search_client.search(search_text=query, select=["id", "content", "metadata"])

    for result in results:
        print(result)


# pure_vector_search
def vector_search(query, clientone, search_client, model, k=3, fields="content_vector"):
    embedding = clientone.embeddings.create(input=query, model=model).data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=k, fields=fields)

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "content", "metadata"],
    )

    for result in results:
        print(f"id: {result['id']}")
        print(f"Score: {result['@search.score']}")
        print(f"content: {result['content']}")
        print(f"metadata: {result['metadata']}\n")


def hybrid_search(query, clientone, search_client, model, k=3, top=3, fields="content_vector"):
    embedding = clientone.embeddings.create(input=query, model=model).data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=k, fields="content_vector")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "metadata"],
        top=top
    )

    for result in results:
        print(f"id: {result['id']}")
        print(f"Score: {result['@search.score']}")
        print(f"content: {result['content']}")
        print(f"metadata: {result['metadata']}\n")


def hyrbid_semantic_search(query, clientone, search_client, model, k=3, top=3, fields="content_vector"):
    embedding = clientone.embeddings.create(input=query, model=azure_openai_embedding_deployment).data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=k, fields="content_vector", exhaustive=True)

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "metadata"],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        top=top
    )

    semantic_answers = results.get_answers()
    for answer in semantic_answers:
        if answer.highlights:
            print(f"Semantic Answer: {answer.highlights}")
        else:
            print(f"Semantic Answer: {answer.text}")
        print(f"Semantic Answer Score: {answer.score}\n")

    for result in results:
        print(f"id: {result['id']}")
        print(f"Score: {result['@search.score']}")
        print(f"Reranker Score: {result['@search.reranker_score']}")
        print(f"content: {result['content']}")
        print(f"metadata: {result['metadata']}")

        captions = result["@search.captions"]
        if captions:
            caption = captions[0]
            if caption.highlights:
                print(f"Caption: {caption.highlights}\n")
            else:
                print(f"Caption: {caption.text}\n")


def delete_documents_by_ids(search_client, document_ids):
    for doc_id in document_ids:
        result_update = search_client.delete_documents(documents=[{"id": doc_id}])


def delete_documents(search_client, document_ids):
    delete_documents_by_ids(search_client, document_ids)


def delete_index(index_name):
    AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)
    client = SearchIndexClient(AZURE_COGNITIVE_SEARCH_SERVICE_NAME, AZURE_COGNITIVE_SEARCH_CREDENTIAL)
    client.delete_index(index_name)


def create_azure_search_index():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_openai_embedding_deployment,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
    )
    AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)
    client = SearchIndexClient(AZURE_COGNITIVE_SEARCH_SERVICE_NAME, AZURE_COGNITIVE_SEARCH_CREDENTIAL)

    embedding_function = embeddings.embed_query

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            retrievable=True,
            analyzer="en.lucene"
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
            analyzer="en.lucene"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=len(embedding_function("Text")),
            vector_search_profile_name="myHnswProfile",
            analyzer="en.lucene"
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
            analyzer="en.lucene"
        ),
    ]

    sc_name = "scoring_profile"
    sc = ScoringProfile(
        name=sc_name,
        text_weights=TextWeights(weights={"content": 5}),
        function_aggregation="sum"
    )

    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    index_name = "testingindex"

    azure_search_instance = AzureSearch(
        azure_search_endpoint=AZURE_COGNITIVE_SEARCH_SERVICE_NAME,
        azure_search_key=AZURE_COGNITIVE_SEARCH_API_KEY,
        vector_search=vector_search,
        index_name=index_name,
        embedding_function=embedding_function,
        semantic_search=semantic_search,
        fields=fields,
        scoring_profiles=[sc],
        default_scoring_profile=sc_name,
        cors_options=cors_options
    )

    return azure_search_instance



create_azure_search_index()
directory = 'dir_name'

# Loading the documents and adding metadata
documents = load_docs(directory)
add_metadata(documents, directory)
max_length = max(len(doc.page_content) for doc in documents)

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=max_length, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(texts)}")

# Adding Documents to the Index using Custom function
upload_doc_to_index()

index_name = "testingindex"
query = "your query?"
clientone = AzureOpenAI(
    azure_deployment=azure_openai_embedding_deployment,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
)
model = azure_openai_embedding_deployment
AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)
search_client = SearchClient(endpoint=AZURE_COGNITIVE_SEARCH_SERVICE_NAME, index_name=index_name,
                             credential=AZURE_COGNITIVE_SEARCH_CREDENTIAL)

get_index(index_name)
keyword_search(search_client, query)
vector_search(query, clientone, search_client, model, k=3, fields="content_vector")
hybrid_search(query, clientone, search_client, model, k=3, top=3, fields="content_vector")
hyrbid_semantic_search(query, clientone, search_client, model, k=3, top=3, fields="content_vector")
document_ids = ["ID1", "ID2"]
delete_documents(search_client, document_ids)
delete_index(index_name)
