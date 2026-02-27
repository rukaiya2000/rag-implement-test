import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DOCUMENTS = [
    "Ragas are melodic frameworks in Indian classical music.",
    "There are many types of ragas, each with its own mood and time of day.",
    "Ragas are used to evoke specific emotions in the listener.",
    "The performance of a raga involves improvisation within a set structure.",
    "Ragas can be performed on various instruments or sung vocally.",
    "Rukaiya is studying ragas to understand their structure and emotional impact.",
    "Rukaiya has basic understanding of ragas but is looking to deepen her knowledge through a RAG application.",
]


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""
    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class EmbeddingsRetriever(BaseRetriever):
    """
    Embeddings-based retriever using OpenAI's embedding models.
    Computes embeddings for documents and queries, then uses cosine similarity
    to find the most semantically relevant documents.
    """

    # FIX 1: Added similarity_threshold parameter so caller can control
    # what counts as "relevant" — critical for embeddings since cosine
    # similarity never returns 0 for unrelated docs (unlike keyword matching)
    def __init__(
        self,
        openai_client: OpenAI,
        embedding_model: str = "sfr-embedding-mistral",
        similarity_threshold: float = 0.5,
    ):
        super().__init__()
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.document_embeddings: List[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using OpenAI's embedding API"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts in a single API call"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [np.array(item.embedding) for item in response.data]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def fit(self, documents: List[str]):
        """Store documents and compute their embeddings"""
        self.documents = documents
        if documents:
            self.document_embeddings = self._get_embeddings_batch(documents)
        else:
            self.document_embeddings = []

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by cosine similarity with query embedding"""
        if not self.documents:
            return []

        query_embedding = self._get_embedding(query)

        scores = []
        for i, doc_embedding in enumerate(self.document_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append((i, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class SimpleKeywordRetriever(BaseRetriever):
    """
    Ultra-simple keyword matching retriever.
    Best used for development, testing, and offline scenarios.
    """

    def __init__(self):
        super().__init__()

    def _count_keyword_matches(self, query: str, document: str) -> int:
        """Count how many query words appear in the document"""
        query_words = query.lower().split()
        document_words = document.lower().split()
        matches = 0
        for word in query_words:
            if word in document_words:
                matches += 1
        return matches

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by keyword match count"""
        scores = []
        for i, doc in enumerate(self.documents):
            match_count = self._count_keyword_matches(query, doc)
            scores.append((i, match_count))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class ExampleRAG:
    """
    RAG system that:
    1. Accepts a pluggable LLM client and retriever
    2. Retrieves relevant documents for a query
    3. Generates a grounded response using the LLM
    4. Traces every step to a log file for observability
    """

    def __init__(
        self,
        llm_client,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        model: str = "gpt-oss-120b",
        logdir: str = "logs",
    ):
        self.llm_client = llm_client
        self.retriever = retriever or SimpleKeywordRetriever()
        # FIX 2: Separated system prompt from user prompt template.
        # Previously system_prompt contained {query} and {context} placeholders
        # and was being sent as BOTH the system message AND formatted into the
        # user message — the LLM was seeing the instructions twice.
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Answer questions using only the provided documents."
        )
        self.user_prompt_template = (
            "Answer the following question based on the provided documents:\n\n"
            "Question: {query}\n\n"
            "Documents:\n{context}\n\n"
            "Answer:"
        )
        # FIX 3: Made model name a parameter instead of hardcoded string
        # so it can be changed without editing the class internals
        self.model = model
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir

        os.makedirs(self.logdir, exist_ok=True)

        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "model": self.model,
                    "logdir": self.logdir,
                },
            )
        )

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base (appends to existing)"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "add_documents",
                    "num_new_documents": len(documents),
                    "total_documents_before": len(self.documents),
                },
            )
        )
        self.documents.extend(documents)
        self.retriever.fit(self.documents)
        self.is_fitted = True
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def set_documents(self, documents: List[str]):
        """Set documents, replacing any existing ones"""
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="rag_system",
                data={
                    "operation": "set_documents",
                    "num_new_documents": len(documents),
                    "old_document_count": len(self.documents),
                },
            )
        )
        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True
        self.traces.append(
            TraceEvent(
                event_type="document_operation",
                component="retriever",
                data={
                    "operation": "fit_completed",
                    "total_documents": len(self.documents),
                    "retriever_type": type(self.retriever).__name__,
                },
            )
        )

    def _get_similarity_threshold(self) -> float:
        """Get the appropriate similarity threshold based on retriever type"""
        # FIX 1 continued: EmbeddingsRetriever uses its own threshold;
        # SimpleKeywordRetriever keeps score > 0 behavior
        if isinstance(self.retriever, EmbeddingsRetriever):
            return self.retriever.similarity_threshold
        return 0.0  # keyword retriever: anything with at least 1 match

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents for the query"""
        if not self.is_fitted:
            raise ValueError(
                "No documents loaded. Call add_documents() or set_documents() first."
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)
        threshold = self._get_similarity_threshold()

        retrieved_docs = []
        for idx, score in top_docs:
            if score > threshold:
                retrieved_docs.append(
                    {
                        "content": self.documents[idx],
                        "similarity_score": score,
                        "document_id": idx,
                    }
                )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "retrieved_docs": [doc["content"] for doc in retrieved_docs],
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    # FIX 4: generate_response now accepts pre-fetched docs instead of
    # re-running retrieval internally. This eliminates the double-retrieval
    # bug where retrieve_documents() was called twice per query() invocation.
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate response using already-retrieved documents.

        Args:
            query: User query
            retrieved_docs: Documents already fetched by retrieve_documents()
        """
        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")
        context = "\n\n".join(context_parts)

        # FIX 2 continued: system prompt is clean instructions,
        # user prompt carries the actual query + context
        user_prompt = self.user_prompt_template.format(query=query, context=context)

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="navigator_api",
                data={
                    "operation": "generate_response",
                    "model": self.model,
                    "query": query,
                    "context": context,
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            response_text = response.choices[0].message.content.strip()

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="navigator_api",
                    data={
                        "operation": "generate_response",
                        "response": response_text,
                        "response_length": len(response_text),
                        "usage": response.usage.model_dump() if response.usage else None,
                        "model": self.model,
                    },
                )
            )

            return response_text

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="navigator_api",
                    data={"operation": "generate_response", "error": str(e)},
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents then generate response.

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID (auto-generated if not provided)
        """
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            # FIX 4 continued: retrieve once, pass docs directly to generate
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(question, retrieved_docs)

            result = {"answer": response, "run_id": run_id}

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {"answer": response, "run_id": run_id, "logs": logs_path}

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={"run_id": run_id, "operation": "query", "error": str(e)},
                )
            )
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export all traces for this run to a JSON log file"""
        timestamp = datetime.now().isoformat()
        log_filename = f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(
    llm_client,
    logdir: str = "logs",
    use_embeddings: bool = True,
) -> ExampleRAG:
    """
    Factory function to create a ready-to-use RAG client.

    Args:
        llm_client: OpenAI-compatible client
        logdir: Directory for trace logs
        use_embeddings: True for semantic search, False for keyword matching
    """
    retriever = (
        EmbeddingsRetriever(openai_client=llm_client)
        if use_embeddings
        else SimpleKeywordRetriever()
    )
    client = ExampleRAG(llm_client=llm_client, retriever=retriever, logdir=logdir)
    client.add_documents(DOCUMENTS)
    return client


if __name__ == "__main__":
    try:
        api_key = os.environ["NAVIGATOR-API-PROJECTS"]
    except KeyError:
        print("Error: NAVIGATOR-API-PROJECTS environment variable is not set.")
        print("export NAVIGATOR-API-PROJECTS='your_navigator_api_key'")
        exit(1)

    llm = OpenAI(
        api_key=api_key,
        base_url="https://api.ai.it.ufl.edu"
    )

    retriever = EmbeddingsRetriever(openai_client=llm)
    rag_client = ExampleRAG(llm_client=llm, retriever=retriever, logdir="logs")
    rag_client.add_documents(DOCUMENTS)

    query = "Is Rag good?"
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Logs: {response['logs']}")