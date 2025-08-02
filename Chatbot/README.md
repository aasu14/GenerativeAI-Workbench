# 🧠 Policy Assistant - AI Chatbot for Organizational Policies

Policy Assistant is an AI-powered chatbot designed to answer questions related to organizational policies, procedures, and guidelines. It leverages Azure OpenAI, LangChain agents, FAISS vector database, and RAG (Retrieval-Augmented Generation) to generate accurate and reference-based responses.

---

## 🚀 Features

- 🔍 Intelligent policy search across multiple departments
- 🧠 RAG-based response generation with source and page number references
- 🧰 Department-specific tools and prompts
- 💬 Chat history with memory
- 💾 JSON logging with token usage stats
- 🙋 Feedback capture mechanism

---

## 🧱 Architecture

- **LangChain**: For tool binding, agent execution, and prompt templating
- **Azure OpenAI**: For embedding and chat completion using GPT-4
- **FAISS**: For vector-based document retrieval
- **Gradio**: For frontend chatbot interface
- **Tiktoken**: For token counting

---

## 🗂️ Directory Structure

```bash
.
├── faiss_index_new/             # FAISS vector index directory
├── final_output.json            # Output log file
├── policy_assistant.py          # Main chatbot logic (code you provided)
├── README.md                    # You're reading this
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/policy-assistant.git
   cd policy-assistant
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Azure OpenAI Environment Variables**
   ```bash
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
   ```

4. **Ensure FAISS Index is Present**
   - Place the FAISS index directory (`faiss_index_new/`) in the root folder.
   - Or generate it using LangChain and save locally.

5. **Run the Application**
   ```bash
   python policy_assistant.py
   ```

---

## 🧪 Example Usage

1. Choose a department from the dropdown.
2. Ask a policy-related question (e.g., *What is the maternity leave policy?*).
3. The bot will:
   - Retrieve relevant policy chunks.
   - Use GPT-4 to answer based on retrieved content.
   - Cite source links and page numbers.
4. Provide feedback via 👍 / 👎 buttons.

---

## 🔁 Core Functionalities

### LLM & Embedding Connection

```python
openai_conn(...)
```
Connects to Azure OpenAI services for embedding and completion.

---

### FAISS Index Loading

```python
faiss_index(...)
```
Loads vector DB with optional HuggingFace embedding fallback.

---

### RAG Pipeline

```python
get_tools(...) → get_executor(...) → chatbot.process_query(...)
```
Retrieves documents → Invokes LLM with tools → Parses final response.

---

### Feedback & Logging

```python
conversation.save_feedback()
conversation.save_json()
```
Captures chat logs, source links, pages, and token usage into `final_output.json`.

---

## 🛡️ Prompt Template

The prompt ensures:
- Answers are based **only** on retrieved policy documents.
- Language is **factual** and **verbatim** from sources.
- Page numbers and source URLs are cited in the response.

---

## 🛠️ Future Improvements

- Integrate user authentication
- Admin interface for uploading/updating policy docs
- Support for multilingual policy documents
- Advanced analytics dashboard for feedback and usage
