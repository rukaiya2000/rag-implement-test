import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from ragas.metrics.collections import ContextPrecision
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
    event_type: str
    component: str
    data: Dict[str, Any]


@dataclass
class EvaluationResult:
    """Holds scores for all three evaluation metrics"""
    context_precision: float
    context_recall: float
    faithfulness: float
    query: str
    answer: str
    retrieved_docs: List[str]
    ground_truth: Optional[str] = None

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"Evaluation Results for query: '{self.query}'\n"
            f"{'='*50}\n"
            f"  Context Precision : {self.context_precision:.3f}\n"
            f"  Context Recall    : {self.context_recall:.3f}\n"
            f"  Faithfulness      : {self.faithfulness:.3f}\n"
            f"{'='*50}"
        )


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using three core metrics:

    1. Context Precision  — Are the retrieved docs actually relevant to the query?
                            (Signal-to-noise ratio of retrieval)

    2. Context Recall     — Do the retrieved docs cover everything needed to
                            answer the query? Requires a ground truth answer.
                            (Coverage of retrieval)

    3. Faithfulness       — Is the generated answer grounded in the retrieved
                            context, or is the LLM hallucinating?
                            (Hallucination detection)

    All three use an LLM-as-judge approach: we ask the LLM to make binary
    yes/no judgements on fine-grained claims, then average the results.
    """

    def __init__(self, llm_client: OpenAI, model: str = "gpt-oss-120b"):
        self.llm_client = llm_client
        self.model = model

    def _parse_yes_no(self, text: str) -> float:
        """Parse a Yes/No LLM response to 1.0 / 0.0."""
        return 1.0 if text.strip().lower().startswith("yes") else 0.0

    # ------------------------------------------------------------------
    # METRIC 1: Context Precision
    # ------------------------------------------------------------------
    async def compute_context_precision(
        self, query: str, retrieved_docs: List[str]
    ) -> float:
        """
        For each retrieved document, ask the LLM: 'Is this doc relevant to the query?'
        Score = (number of relevant docs) / (total retrieved docs)

        High precision = most retrieved docs are actually useful.
        Low precision  = lots of noise / off-topic chunks retrieved.
        """
        scorer = ContextPrecision(llm=llm)
        if not retrieved_docs:
            return 0.0

        relevant_count = 0
        for doc in retrieved_docs:
            
            result = await scorer.ascore(
            user_input=query,
            reference="",
            retrieved_contexts=[
                doc
            ]
        )
            relevant_count += self._parse_yes_no(result)

        return relevant_count / len(retrieved_docs)

    # ------------------------------------------------------------------
    # METRIC 2: Context Recall
    # ------------------------------------------------------------------
    async def compute_context_recall(
        self, query: str, retrieved_docs: List[str], ground_truth: str
    ) -> float:

        if not retrieved_docs or not ground_truth:
            return 0.0

        # Step 1: Extract atomic claims from the ground truth
        claims_prompt = (
            f"Break the following answer into a list of independent factual claims.\n"
            f"Return each claim on a new line, numbered (1. 2. 3. ...).\n\n"
            f"Answer: {ground_truth}"
        )
       
        # Parse numbered lines into individual claims
        claims = []
        for line in claims_response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Strip leading "1. ", "2. " etc.
                claim = line.split(".", 1)[-1].strip()
                if claim:
                    claims.append(claim)

        if not claims:
            return 0.0

        context = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)]
        )

        # Step 2: For each claim, check if it's supported by the context
        supported = 0
        for claim in claims:
            prompt = (
                f"Context documents:\n{context}\n\n"
                f"Claim: {claim}\n\n"
                "Can this claim be supported or inferred from the context documents above? "
                "Answer with only 'Yes' or 'No'."
            )
            result = self._llm_judge(prompt)
            supported += self._parse_yes_no(result)

        return supported / len(claims)

    # ------------------------------------------------------------------
    # METRIC 3: Faithfulness
    # ------------------------------------------------------------------
    async def compute_faithfulness(
        self, query: str, answer: str, retrieved_docs: List[str]
    ) -> float:
        """
        Break the generated answer into individual claims, then for each claim
        ask: 'Is this claim supported by the retrieved context?'
        Score = (supported claims) / (total claims)

        High faithfulness = answer sticks to what the docs say (no hallucination).
        Low faithfulness  = LLM is making things up beyond the retrieved context.
        """
        if not answer or not retrieved_docs:
            return 0.0

        # Step 1: Extract atomic claims from the generated answer
        claims_prompt = (
            f"Break the following answer into a list of independent factual claims.\n"
            f"Return each claim on a new line, numbered (1. 2. 3. ...).\n\n"
            f"Answer: {answer}"
        )
        claims_response = self._llm_judge(claims_prompt)

        claims = []
        for line in claims_response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                claim = line.split(".", 1)[-1].strip()
                if claim:
                    claims.append(claim)

        if not claims:
            return 0.0

        context = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)]
        )

        # Step 2: For each claim, verify it's grounded in the context
        supported = 0
        for claim in claims:
            prompt = (
                f"Context documents:\n{context}\n\n"
                f"Claim: {claim}\n\n"
                "Is this claim fully supported by the context documents above? "
                "Answer with only 'Yes' or 'No'."
            )
            result = self._llm_judge(prompt)
            supported += self._parse_yes_no(result)

        return supported / len(claims)

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run all three metrics and return an EvaluationResult.

        Args:
            query:          The user's question
            answer:         The generated answer from the RAG pipeline
            retrieved_docs: The context chunks that were retrieved
            ground_truth:   Optional reference answer (needed for recall)
        """
        print("  [eval] computing context precision...")
        precision = self.compute_context_precision(query, retrieved_docs)

        print("  [eval] computing context recall...")
        recall = (
            self.compute_context_recall(query, retrieved_docs, ground_truth)
            if ground_truth
            else 0.0
        )

        print("  [eval] computing faithfulness...")
        faithfulness = self.compute_faithfulness(query, answer, retrieved_docs)

        return EvaluationResult(
            context_precision=precision,
            context_recall=recall,
            faithfulness=faithfulness,
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            ground_truth=ground_truth,
        )


# ======================================================================
# Rest of your existing classes below — unchanged except query() which
# now optionally accepts ground_truth and runs evaluation at the end
# ======================================================================

class BaseRetriever:
    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        raise NotImplementedError


class EmbeddingsRetriever(BaseRetriever):
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
        response = self.openai_client.embeddings.create(
            model=self.embedding_model, input=text
        )
        return np.array(response.data[0].embedding)

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model, input=texts
        )
        return [np.array(item.embedding) for item in response.data]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def fit(self, documents: List[str]):
        self.documents = documents
        self.document_embeddings = (
            self._get_embeddings_batch(documents) if documents else []
        )

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        if not self.documents:
            return []
        query_embedding = self._get_embedding(query)
        scores = [
            (i, self._cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.document_embeddings)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class SimpleKeywordRetriever(BaseRetriever):
    def _count_keyword_matches(self, query: str, document: str) -> int:
        query_words = set(query.lower().split())
        document_words = set(document.lower().split())
        return len(query_words & document_words)

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        scores = [
            (i, self._count_keyword_matches(query, doc))
            for i, doc in enumerate(self.documents)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class ExampleRAG:
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
        self.model = model
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir
        self.evaluator = RAGEvaluator(llm_client=llm_client, model=model)
        os.makedirs(self.logdir, exist_ok=True)

    def add_documents(self, documents: List[str]):
        self.documents.extend(documents)
        self.retriever.fit(self.documents)
        self.is_fitted = True

    def set_documents(self, documents: List[str]):
        self.documents = documents
        self.retriever.fit(self.documents)
        self.is_fitted = True

    def _get_similarity_threshold(self) -> float:
        if isinstance(self.retriever, EmbeddingsRetriever):
            return self.retriever.similarity_threshold
        return 0.0

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise ValueError("No documents loaded.")

        top_docs = self.retriever.get_top_k(query, k=top_k)
        threshold = self._get_similarity_threshold()

        return [
            {"content": self.documents[idx], "similarity_score": score, "document_id": idx}
            for idx, score in top_docs
            if score > threshold
        ]

    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        context = "\n\n".join(
            [f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)]
        )
        user_prompt = self.user_prompt_template.format(query=query, context=context)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def query(
        self,
        question: str,
        top_k: int = 3,
        run_id: Optional[str] = None,
        ground_truth: Optional[str] = None,   # <-- NEW: pass for recall scoring
        run_evaluation: bool = True,            # <-- NEW: set False to skip eval
    ) -> Dict[str, Any]:
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        retrieved_docs = self.retrieve_documents(question, top_k)
        answer = self.generate_response(question, retrieved_docs)

        result: Dict[str, Any] = {"answer": answer, "run_id": run_id}

        if run_evaluation:
            print(f"\nRunning evaluation for run_id={run_id}...")
            doc_texts = [d["content"] for d in retrieved_docs]
            eval_result = self.evaluator.evaluate(
                query=question,
                answer=answer,
                retrieved_docs=doc_texts,
                ground_truth=ground_truth,
            )
            print(eval_result.summary())
            result["evaluation"] = {
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "faithfulness": eval_result.faithfulness,
            }

        return result


def default_rag_client(
    llm_client, logdir: str = "logs", use_embeddings: bool = True
) -> ExampleRAG:
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
        exit(1)

    llm = OpenAI(api_key=api_key, base_url="https://api.ai.it.ufl.edu")

    retriever = EmbeddingsRetriever(openai_client=llm)
    rag_client = ExampleRAG(llm_client=llm, retriever=retriever, logdir="logs")
    rag_client.add_documents(DOCUMENTS)

    query = "What are ragas used for?"
    ground_truth = "Ragas are melodic frameworks used in Indian classical music to evoke specific emotions in the listener."

    response = rag_client.query(query, top_k=3, ground_truth=ground_truth)
    print("\nAnswer:", response["answer"])
    print("\nEvaluation scores:", response["evaluation"])