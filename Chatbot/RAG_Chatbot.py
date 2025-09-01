import os
import time
import timeit
import json
import tiktoken
import gradio as gr
from typing import List
from datetime import datetime
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document


def openai_conn(openai_version, embed_model, openai_model, openai_model_name, openai_keys, openai_endpoint, temperature,
                top_p, seed):
    os.environ["AZURE_OPENAI_API_KEY"] = openai_keys
    os.environ["AZURE_OPENAI_ENDPOINT"] = openai_endpoint
    embedding_model = AzureOpenAIEmbeddings(azure_deployment=embed_model, openai_api_version=openai_version)
    llm = AzureChatOpenAI(deployment_name=openai_model, model_name=openai_model_name,
                          openai_api_version=openai_version, temperature=temperature, top_p=top_p, seed=seed)

    return embedding_model, llm


def faiss_index(embedding_model, faiss_path, hf_embeddings=False):
    if hf_embeddings:
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.load_local(f"{faiss_path}", embedding_model, allow_dangerous_deserialization=True)
    return db


def get_retriever(db, search_type, score_threshold, docs_retriever_k, department):
    retriever = doc_score_retriever(vectorstore=db, search_type=search_type,
                                    search_kwargs={"score_threshold": score_threshold, "k": docs_retriever_k,
                                                   'filter': {'departments': f'{department}'}},
                                    chain_type="stuff")
    return retriever


def get_tools(db, search_type, score_threshold, docs_retriever_k, department):
    retriever = get_retriever(db, search_type, score_threshold, docs_retriever_k, department)

    if department == 'department1':
        tool_desc = """department_desc"""

    elif department == 'department2':
        tool_desc = """department2_desc"""

    elif department == 'department3':
        tool_desc = """department3_desc"""

    elif department == 'department4':
        tool_desc = """department4_desc"""

    elif department == 'department5':
        tool_desc = """department5_desc"""

    else:
        return []

    return [Tool(func=retriever.invoke, description=tool_desc, name=f'Policy-{department}')]


class Response(BaseModel):
    output: str = Field(description="The final answer to respond to the user")
    sources: List[str] = Field(
        description="""List of links that are used for generating the final answer to the question.
            Only include a links if it contains relevant information""")
    pages: str = Field(description="""Identify the *** Page Identification number : *** from the provided 
    documents that are used to generate the final response..""")


def parse(output):
    # If no function was invoked, return to user
    # Input to parse - LLM output - is of type AIMessageChunk = Content + Additional Kwargs
    # Output of parse - Return is of type = AgentAction/AgentFinish
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        # AgentActionMessageLog = AgentAction + Message Log. Extension of AgentAction.
        # AgentAction = Tool + Tool Input + Log
        # OpenAIToolAgentAction is an AgentAction. AgentAction = Tool + Tool inputs + Log + message_log
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )


def get_prompt():
    MEMORY_KEY = "chat_history"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert in organizational policies. Your primary role is to meticulously review 
            documents related to company policies, procedures, and guidelines. You provide clear, accurate, and 
            comprehensive answers to any questions from these documents."""),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", """Based on the provided organizational policies, provide the answers of the question:
            {input}
            Be detailed and exhaustive. Use exact language as found in the provided organizational policies. 
            The provided organizational policies are your only source of truth. 
            Please ensure that all numbers and values are accurate and carefully verified before answering 
            the question within the context."""),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return prompt


def get_executor(prompt, llm_with_tools, tools, parse):
    # Agent = Prompt mapping + Prompt + LLM bound with tools + Output Parser
    agent = (
            {
                "input": lambda x: x["input"],
                # Format agent scratchpad from intermediate steps
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | parse
    )
    agent_executor = AgentExecutor(tools=tools, agent=agent, verbose=True, max_iterations=2,
                                   return_intermediate_steps=True)

    # Code below is good way to understand AgentExecutor
    #   next_action = agent.get_action(...)
    # max_iterations is the number of runs of this loop below.
    #   while next_action != AgentFinish:
    #        # Each line of code below generates an output i.e message that is added into the prompt (FunctionMessage/AIMessage) + Agent Step/Agent Action
    #
    #       What does the line below do? ---> 1) Current Agent Action generates an Output. Output = FunctionMessage
    #                                       2) AgentStep is also created = Current AgentAction + Observation (output of Current Agent Action).
    #                                       3) No LLM hit/ No output parser. Just the tool.
    #       observation = run(next_action)
    #
    #       What does the line below do? ---> 1) Current Agent Action + Observation generates an Output. Output = AIMessage
    #                                       2) Next AgentAction is also created using = Current AgentAction + Observation
    #                                       3) LLM hit and output parser called
    #       next_action = agent.get_action(..., next_action, observation)
    # return next_action

    return agent_executor


def write_to_json(data, filename):
    with open(filename, 'a') as f:
        json.dump(data, f)
        f.write('\n')


class Chatbot:
    def __init__(self, departments=''):
        self.state = "query"
        self.departments = departments
        self.default_count = 0
        self.last_query_time = time.time()
        self.chat_history = []

        # 0.5 across all depts
        self.threshold_retriever = db.as_retriever(search_type="similarity_score_threshold",
                                                   search_kwargs={"k": 1, "score_threshold": 0.5})
        # 0.65 across 1 dept
        self.threshold_retriever_query = db.as_retriever(search_type="similarity_score_threshold",
                                                         search_kwargs={"k": 1, "score_threshold": 0.65,
                                                                        'departments': f'{self.departments}'})

        tools = get_tools(db, search_type, score_threshold, docs_retriever_k, self.departments)
        if tools:
            # LLM uses index with 0.5 threshold and dept filter
            scratch_llm = llm.bind_functions([tools[0], Response])
            self.conversational_agent = get_executor(prompt, scratch_llm, tools, parse)

    def process_query(self, query, conversation):
        response = None
        with CodeTimer('process_query'):
            if self.state == "welcome":
                self.state = "query"
                response = Welcome_Message
            elif self.state == "query":
                self.last_query_time = time.time()
                response = self.answer_state(query, conversation)
                self.state = "query"  # Transition back to query state
            elif self.state == "default":
                response = self.default_state(query, conversation)
                self.state = "query"  # Transition back to query state
            elif self.state == "exit":
                response = self.exit_state(query, conversation)
        return response

    def get_rag_response(self, query, conversation):
        with CodeTimer('Invoking agent with query and chat history'):
            response = self.conversational_agent.invoke({"input": query, "chat_history": self.chat_history})

        with CodeTimer('Formatting response & building chat history'):
            if (response['intermediate_steps']) and (response['sources']):
                final_response = response['output'] + '\n'
                ret_sources = list([doc.metadata['link'] for doc in response['intermediate_steps'][0][1]])
                for llm_use_sources in response['sources']:
                    doc_id = llm_use_sources.split('=')[1]
                    page_number = ",".join([s.replace(f"{doc_id}_", "")
                                            for s in response['pages'].split(", ") if s.startswith(f"{doc_id}_")])
                    final_response += (f"\n Content retrieved from Page <b>{page_number}</b> of the "
                                       f"policy id <b>{doc_id}</b>.\n Link: <b>{llm_use_sources}</b>")

                conversation.set_input(query)
                conversation.set_response(response)
                conversation.set_reference(str(ret_sources))

                # build chat history
                self.chat_history.extend([HumanMessage(content=query), AIMessage(content=response['output'])])
                if len(self.chat_history) > 8: [self.chat_history.pop(0) for _ in range(2)]
                self.default_count = 0
                return final_response
        return None

    def answer_state(self, query, conversation):
        if llm.invoke(Thanks_Prompt.format(prompt=query)).content == "category_1":
            return self.exit_state(query, conversation)
        else:
            with CodeTimer('Initial Retrieval - 0.5'):
                # Search across all departments with threshold 0.5
                if not self.threshold_retriever.invoke(query):
                    # No point hitting LLM
                    return self.default_state(query, conversation)

            with CodeTimer('Getting RAG Response for the user query from all departments'):
                result = self.get_rag_response(query, conversation)
            if result is None:
                # No documents used from that dept with threshold > 0.5 or none got used
                # Search across specific departments with threshold 0.65
                with CodeTimer('2nd Retrieval - 0.65'):
                    if not self.threshold_retriever_query.invoke(query):
                        return self.default_state(query, conversation)
                with CodeTimer('Getting RAG Response from specific department'):
                    result = self.get_rag_response(str(query) + '? must use the documents.', conversation)
            return result if result is not None else self.default_state(query, conversation)

    def default_state(self, query, conversation):
        conversation.set_input(query)
        conversation.set_unanswered()
        self.default_count += 1
        if self.default_count >= 3:
            return self.exit_state(query, conversation)
        else:
            self.state = "query"
            return Default_Message

    def exit_state(self, query, conversation):
        conversation.set_input(query)
        conversation.set_unanswered()
        if self.default_count >= 3:
            self.state == "welcome"
            exit_res = FAQ
        else:
            exit_res = Exit_Message
        self.default_count = 0
        return exit_res


class doc_score_retriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs))
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity
        docs = [doc for doc, _ in docs_and_similarities]
        return docs


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        print(f'Code block{self.name} took: {self.took:.7f} s')


class Conversation():

    def __init__(self):
        self.input = ''
        self.response = ''
        self.reference = ''
        self.reference_used = ''
        self.chat_time = ''
        self.timestamp = datetime.now()
        self.feedback = ''
        self.department = ''
        self.answered = 'Yes'
        self.start_time = time.time()
        self.chat_history = ''
        self.intermediate_steps = ''
        self.output = ''
        self.pages = ''

    def set_input(self, input):
        self.input = input

    def set_response(self, response):
        self.response = response

    def set_reference(self, reference):
        self.reference = reference

    def set_department(self, department):
        self.department = department

    def set_unanswered(self):
        self.answered = 'No'

    def set_response_variables(self):
        self.chat_history = str(self.response['chat_history'])
        self.intermediate_steps = str(self.response['intermediate_steps'])
        self.reference_used = str(self.response['sources'])
        self.output = str(self.response['output'])
        self.pages = str(self.response['pages'])

    def save_json(self):
        response_time = time.time() - self.start_time

        if self.answered == 'Yes':
            self.set_response_variables()

        input_prompt = self.input + self.chat_history + self.intermediate_steps + self.reference

        prompt_tokens = self.token_usage(input_prompt)
        completion_tokens = self.token_usage(self.output)

        data = {'input': self.input, 'output': self.output, 'reference_retrival': str(self.reference),
                'reference_used': self.reference_used, 'chat_time': response_time,
                'timestamp': str(self.timestamp), 'department': self.department, 'feedback': self.feedback,
                'answered': self.answered, 'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens, 'page_number': self.pages}
        write_to_json(data, 'final_output.json')

    def save_feedback(self, history, feedback):
        with open('final_output.json', 'r') as f:
            data = f.read().strip().split('\n')

        # Initialize the list to store modified JSON objects
        modified_data = []
        history = ','.join(str(v) for v in history)
        # Process each JSON string
        for jsonstr in data:
            out = json.loads(jsonstr)
            time_taken = (datetime.now() - datetime.fromisoformat(out['timestamp'])).seconds
            if (out['input'].replace('? must use the documents.', '') in history) and (time_taken < 120):
                out['feedback'] = feedback
            modified_data.append(out)

            # Write the modified JSON objects back to the file
            with open('final_output.json', 'w') as f:
                for item in modified_data:
                    f.write(json.dumps(item) + '\n')

    def token_usage(self, string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


if __name__ == '__main__':

    Welcome_Message = "Hello and welcome to **Policy Assistant**! You can ask me questions related to standard"

    Exit_Message = "Thanks for using **Policy Assistant**! For future queries, feel free to come back anytime. " \
                   "\n You can also provide feedback using the thumbs up/down button and clicking on Submit. " \
                   "Your input matters. \n\n Have a great day and goodbye for now!"

    Default_Message = "I'm currently unable to find the information you're seeking. Can you please" + \
                      " rephrase your question or provide more details? \n\n Alternatively, you may wish to contact " + \
                      "_**xx@email.com**_ directly for further assistance."

    FAQ = "It seems we're having a bit of trouble finding the answers to your questions. For frequently asked " + \
          "questions, you can check our **FAQ section**. \n\n If you have specific questions about SOPs or policies, " + \
          "feel free to ask me again or reach out to _**xx@email.com**_ for personalized assistance."

    Thanks_Prompt = """
    **Query**: {prompt}

    **Objective**: 
    Below is a dictionary provided consisting of a category key and value denoting the intent. Can you check 
    If the above query belongs to one of the category keys based on the intent specified.

    dictionary = {{'category_1':'thank you or okay state','category_2':'query state'}}

    **Note**: 
    Please respond based on the Query. The response should only be the category key.
    """

    openai_version = "2024-02-15-preview"
    embed_model = "Ada-Embedding"
    openai_model = "GPT_4_Omini"
    openai_model_name = "gpt-4"
    openai_keys = ""
    openai_endpoint = ""

    temperature = 0.5
    top_p = 0.4
    seed = 42
    faiss_index_path = r"faiss_index_new/"

    search_type = "similarity_score_threshold"
    score_threshold = 0.5
    docs_retriever_k = 2

    embedding_model, llm = openai_conn(openai_version, embed_model, openai_model, openai_model_name, openai_keys,
                                       openai_endpoint, temperature, top_p, seed)
    db = faiss_index(embedding_model, faiss_index_path)
    prompt = get_prompt()
    chatbot = Chatbot()

    prev_departments = ['empty_department']


    def predict(message, history, Departments, Feedback):
        feedback = Feedback
        departments = Departments
        conversation = Conversation()
        conversation.set_department(departments)
        if feedback and not message:
            if history:
                conversation.save_feedback(history, feedback)
                return "Thank you for your feedback. \n\n Please clear the chat and start again."

            else:
                return "No conversation has occurred yet. Please ask your question first, and then provide the feedback."

        if not departments:
            return ("Hello! Kindly choose the department you wish to inquire about from the dropdown under "
                    "'Additional Inputs' section below.")
        else:
            if not history:
                chatbot.__init__(departments)
                prev_departments.clear()
            try:
                if prev_departments[-1] != departments:
                    chatbot.__init__(departments)
                    prev_departments.append(departments)
            except:
                prev_departments.append(departments)

            try:
                rag_response = chatbot.process_query(message, conversation)
                if rag_response == Exit_Message or rag_response == FAQ:
                    chatbot.__init__(departments)

            except Exception as e:
                rag_response = "Oops, something went wrong!"
                chatbot.__init__(departments)

        conversation.save_json()
        return rag_response


    welcome_greet = gr.Chatbot(value=[(None, Welcome_Message)], elem_id="ChatInterface", scale=2)
    drop_menu = gr.Dropdown(["department1", "department2", "department3", "department4", "department5"],
                            label="Departments", info="Kindly choose the department")
    feedback_menu = gr.Radio(["👍", "👎"], label="Feedback", info="Please share your valuable feedback")

    gr.ChatInterface(fn=predict, additional_inputs=[drop_menu, feedback_menu],
                     chatbot=welcome_greet).launch(server_name='0.0.0.0')
