# LLM Response Evaluator

This project evaluates the **relevance** and **completeness** of answers generated by a language model against a **ground truth**, using OpenAI's GPT models. It uses a FAISS vector store to retrieve the context (ground truth) and scores responses based on how well they align with the expected answer.

---

## 🧠 Features

- Scores answers based on **relevance** (1–10) and **completeness** (0–100)
- Uses **FAISS** for document retrieval
- Supports **OpenAI GPT models** via a modular `GenOpenai` interface
- Evaluates from JSON lines input file

---

## 📁 Project Structure

```bash
.
├── evaluate_responses.py     # Main script to run evaluation
├── openai_config.py          # OpenAI model deployment config
├── final_output.json         # Input/output file (JSONL format)
└── README.md                 # This file
```

---

## 🛠️ Requirements

Install required libraries:

```bash
pip install langchain openai faiss-cpu
```

Also, make sure to include your custom module:
- `AIQ_Common.util.openai_handler` with a `GenOpenai` class
- `openai_config.py` containing model names and parameters like `OPEN_AI_MODELS_PARAMS` and `PARAMETERS`

---

## ⚙️ Configuration

### `openai_config.py` Example

```python
OPEN_AI_MODELS_PARAMS = {
    'GPT-4-Omini': {
        'deployment_name': 'gpt-4-omni-deployment',
    }
}

PARAMETERS = {
    'temperature': 0.3,
    'max_tokens': 100,
    ...
}
```

---

## 📄 Input Format (`final_output.json`)

The input should be in **JSONL** format (`\n`-separated JSON strings), with each object having:

```json
{
  "input": "What is the capital of France?",
  "output": "Paris is the capital of France.",
  "reference_used": ["http://example.com/source1"],
  "answered": "Yes"
}
```

---

## 🚀 How to Run

```bash
python evaluate_responses.py
```

This will:
1. Load the FAISS index.
2. Iterate through each line in the `final_output.json` file.
3. For each answered question, extract relevant content based on references.
4. Generate relevance and completeness scores using GPT model.
5. Overwrite the original file with updated scores.

---

## 📝 Output Example

Each line in the output file will now include:

```json
{
  "input": "...",
  "output": "...",
  "reference_used": [...],
  "answered": "Yes",
  "relevance_score": "9",
  "completeness_score": "95"
}
```

---

## 🔒 Notes

- Ensure that the FAISS index at `index_name/` is compatible with your embedding model.
- `allow_dangerous_deserialization=True` is used — make sure the source is trusted.
- The code assumes that every document in FAISS has a `metadata['link']`.
