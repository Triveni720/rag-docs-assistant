"""Retrieval-Augmented Generation (RAG) over a folder of Markdown documents.

Retrieval uses TF-IDF vectors and cosine similarity (scikit-learn).
Generation uses Google Gemini when GEMINI_API_KEY is set; otherwise the
assistant returns the best matching passage (extractive mode), so the
project also runs and tests without an API key.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Chunk:
    source: str   # file name, e.g. "kubernetes.md"
    section: str  # heading the text came from
    text: str


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    """Split every Markdown file into one chunk per section."""
    chunks = []
    for path in sorted(data_dir.glob("*.md")):
        section, lines = "Introduction", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                if "".join(lines).strip():
                    chunks.append(Chunk(path.name, section, " ".join(lines).strip()))
                section, lines = line.lstrip("#").strip(), []
            elif line.strip():
                lines.append(line.strip())
        if "".join(lines).strip():
            chunks.append(Chunk(path.name, section, " ".join(lines).strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(
            [f"{c.section} {c.text}" for c in chunks]
        )

    def search(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k] if score > 0]


def build_prompt(question: str, contexts: list[Chunk]) -> str:
    context_text = "\n\n".join(f"[{c.source} - {c.section}] {c.text}" for c in contexts)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )


def call_gemini(prompt: str) -> str:
    """Call the Gemini API. Imported lazily so tests don't need the SDK or a key."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt
    )
    return response.text.strip()


class Assistant:
    def __init__(self, retriever: Retriever, llm=None):
        self.retriever = retriever
        if llm is None and os.getenv("GEMINI_API_KEY"):
            llm = call_gemini
        self.llm = llm  # None = extractive mode

    def ask(self, question: str, k: int = 3) -> dict:
        results = self.retriever.search(question, k=k)
        if not results:
            return {"answer": "I don't know - no relevant documents found.", "sources": [], "mode": "none"}
        contexts = [chunk for chunk, _ in results]
        sources = [
            {"source": c.source, "section": c.section, "score": round(s, 3)} for c, s in results
        ]
        if self.llm:
            answer, mode = self.llm(build_prompt(question, contexts)), "llm"
        else:
            answer, mode = contexts[0].text, "extractive"
        return {"answer": answer, "sources": sources, "mode": mode}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
