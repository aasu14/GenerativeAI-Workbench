import json
from langchain_community.vectorstores import FAISS
from openai_handler import GenOpenai
from openai_config import *


RELEVANCE_PROMPT = """
Your role is to assess the relevance of the answer given in response to a particular question, using the 
'ground truth' or actual fact as your benchmark. First, please carefully read the 
'ground truth' which is the actual fact or correct answer. Then, read the posed question and the given answer. 
Your task is to judge how relevant or accurate the provided answer is in relation to the question, 
based on the ground truth information. Here are the details you need to consider: 
Ground Truth: {page_content} 
This is the factual information against which you will assess the answer's relevance. 
Question: {question}.
This is the query to which the answer was provided. 
Answer: {answer}.
This is the response that needs to be evaluated for its relevance to the question and accuracy against the ground truth. 
Upon reviewing these, please provide a relevance score on a scale from 1 to 10. A score of 1 indicates 
that the answer is not relevant at all to the question or the ground truth, while a score of 10 means the answer 
is highly relevant and accurately reflects the ground truth. Please provide response that contain only relevance score.
"""

COMPLETENESS_PROMPT = """
Your responsibility here is to determine the completeness of an answer provided in response to a specific question. 
You will be given three pieces of information: the ground truth, which is the ideal, comprehensive, 
and correct answer to the question; the question that was asked; and the answer that was initially provided.
Here's the details you'll be working with: 
Ground Truth: {page_content} 
Question: {question} 
Answer: {answer}. 
Your task is to provide a completeness score for the original answer. This score should be on a scale from 0 to 100, 
where 0 indicates that the original answer provided is completely off the mark or doesn't answer the question at all, 
and 100 indicates that the original answer is entirely complete and accurate, fully addressing the question and 
aligning with the ground truth. While evaluating the completeness of the answer, consider the following: 
Does the answer address all aspects of the question? Is the information in the answer accurate and does 
it align with the ground truth? Please provide response that contain only the completeness score as a number.
"""


def evaluate_result(model_name, question, page_content, answer, evaluate_type):
    if evaluate_type == 'completeness':
        PROMPT = COMPLETENESS_PROMPT.format(question=question, page_content=page_content, answer=answer)
    elif evaluate_type == 'relevance':
        PROMPT = RELEVANCE_PROMPT.format(question=question, page_content=page_content, answer=answer)
    else:
        PROMPT = ''

    messages = genai.get_api_message(user_message=PROMPT)
    response = genai.concurrent_fetch_gpt_response(OPEN_AI_MODELS_PARAMS[model_name]['deployment_name'], messages,
                                                   PARAMETERS)
    return response['choices'][0]['message']['content']


def compute_evaluation(filename, model_name):
    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')

    modified_data = []
    for jsonstr in data:
        page_content = ''
        out = json.loads(jsonstr)
        if out['answered'] == 'Yes':
            question = out['input']
            answer = out['output']
            references = out['reference_used']
            for doc in db.docstore._dict.values():
                if doc.metadata['navex_link'] in references:
                    page_content = page_content + doc.page_content

            if page_content is not None:
                out['relevance_score'] = evaluate_result(model_name, question, page_content, answer, 'relevance')
                out['completeness_score'] = evaluate_result(model_name, question, page_content, answer, 'completeness')
                modified_data.append(out)

    # Write the modified JSON objects back to the file
    with open(filename, 'w') as f:
        for item in modified_data:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    embedding_model = 'Ada-Embedding'
    llm_model = 'GPT-4-Omini'
    faiss_index_path = r"index_name/"
    output_file = 'final_output.json'

    embeddings = GenOpenai(model_name=embedding_model)
    embedding_model = embeddings.get_embedding_model()
    db = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

    genai = GenOpenai(model_name=llm_model)
    compute_evaluation(filename=output_file, model_name=llm_model)
