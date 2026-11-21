# RAG Docs Assistant

![CI](https://github.com/Triveni720/rag-docs-assistant/actions/workflows/ci.yml/badge.svg)

A Retrieval-Augmented Generation (RAG) assistant that answers questions about MuleSoft, Salesforce and Kubernetes documents. It retrieves the most relevant sections, then asks Google Gemini to answer using only that context, and returns the sources it used.

## How it works

1. **Chunking:** each Markdown file in `data/` is split into one chunk per section.
2. **Retrieval:** chunks are turned into TF-IDF vectors (scikit-learn) and ranked by cosine similarity to the question.
3. **Generation:** the top chunks go into a grounded prompt ("answer using ONLY the context") sent to the Gemini API.
4. **Fallback:** without an API key, it returns the best matching passage (extractive mode), so tests run offline.

## Evaluation

`evaluate.py` scores retrieval against a golden dataset of 12 questions (`tests/golden.json`):

| Metric | Result |
| --- | --- |
| Hit rate @3 | 1.0 |
| Mean reciprocal rank (MRR) | 1.0 |

CI fails if hit rate drops below 0.9 or MRR below 0.75.

## Tech stack

Python, scikit-learn, Google Gemini API, FastAPI, pytest, Docker, GitHub Actions

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # optional, free key from aistudio.google.com
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /ask` with `{"question": "What happens if a pod is deleted?"}`.

## Tests and evaluation

```bash
pytest -v
python evaluate.py
```

## Docker

```bash
docker build -t rag-docs-assistant .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" rag-docs-assistant
```

## Author

Triveni Battula
