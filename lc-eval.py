from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langsmith import traceable

# Load environment variables from .ENV file
load_dotenv()

api_key = os.environ["NAVIGATOR-API-PROJECTS"]

# LangSmith configuration — set these in your .env file:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=<your-langsmith-api-key>
# LANGCHAIN_PROJECT=rag-evaluation  (optional, defaults to "default")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "rag-evaluation")


docs = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
    Document(page_content="It enables applications that use a language model as reasoning element.", metadata={"source": "doc2"}),
    Document(page_content="RAG combines retrieval of relevant documents with generation using LLMs.", metadata={"source": "doc3"}),
    Document(page_content="LangChain supports multiple LLM providers including OpenAI, Anthropic, Cohere, and open-source models via Hugging Face.", metadata={"source": "doc4"}),
    Document(page_content="Chains in LangChain allow you to combine multiple components such as prompts, LLMs, and output parsers into a single pipeline.", metadata={"source": "doc5"}),
    Document(page_content="Agents in LangChain use LLMs to decide which tools to call and in what order to complete a task.", metadata={"source": "doc6"}),
    Document(page_content="Memory components in LangChain allow conversational applications to retain context across multiple turns of dialogue.", metadata={"source": "doc7"}),
    Document(page_content="Vector stores such as Chroma, Pinecone, and FAISS are used in LangChain to store and retrieve document embeddings efficiently.", metadata={"source": "doc8"}),
    Document(page_content="Embeddings convert text into dense numerical vectors that capture semantic meaning, enabling similarity search.", metadata={"source": "doc9"}),
    Document(page_content="LangSmith is a platform for debugging, testing, evaluating, and monitoring LangChain applications in production.", metadata={"source": "doc10"}),
    Document(page_content="Retrieval-Augmented Generation (RAG) improves LLM accuracy by grounding responses in externally retrieved, up-to-date information.", metadata={"source": "doc11"}),
    Document(page_content="LangChain Expression Language (LCEL) provides a declarative way to compose chains using the pipe operator.", metadata={"source": "doc12"}),
    Document(page_content="Output parsers in LangChain transform the raw text output of an LLM into structured formats such as JSON, lists, or Pydantic models.", metadata={"source": "doc13"}),
]

# ─────────────────────────────────────────────
# Text splitting
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)



# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="sfr-embedding-mistral",
    openai_api_base="https://api.ai.it.ufl.edu",
    api_key=api_key,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False
)

# ─────────────────────────────────────────────
# Chroma vector store
# ─────────────────────────────────────────────
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    openai_api_base="https://api.ai.it.ufl.edu",
    model="llama-3.1-70b-instruct",
    api_key=api_key,
    temperature=0.1
)

# ─────────────────────────────────────────────
# RAG chain
# ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Wrap the chain so LangSmith traces each call
@traceable(name="rag_pipeline")
def run_rag(question: str) -> str:
    return rag_chain.invoke(question)

# ─────────────────────────────────────────────
# LangSmith Evaluation
# ─────────────────────────────────────────────

# 10 test queries with reference (ground-truth) answers
EVAL_DATASET = [
    {
        "question": "What is LangChain?",
        "answer": "LangChain is a framework for developing applications powered by language models."
    },
    {
        "question": "What is RAG?",
        "answer": "RAG (Retrieval-Augmented Generation) combines retrieval of relevant documents with generation using LLMs to improve accuracy."
    },
    {
        "question": "What LLM providers does LangChain support?",
        "answer": "LangChain supports OpenAI, Anthropic, Cohere, and open-source models via Hugging Face."
    },
    {
        "question": "What are agents in LangChain?",
        "answer": "Agents use LLMs to decide which tools to call and in what order to complete a task."
    },
    {
        "question": "What is LangSmith used for?",
        "answer": "LangSmith is used for debugging, testing, evaluating, and monitoring LangChain applications in production."
    },
    {
        "question": "What are vector stores and give some examples?",
        "answer": "Vector stores store and retrieve document embeddings. Examples include Chroma, Pinecone, and FAISS."
    },
    {
        "question": "What are embeddings?",
        "answer": "Embeddings convert text into dense numerical vectors that capture semantic meaning, enabling similarity search."
    },
    {
        "question": "What is LCEL?",
        "answer": "LangChain Expression Language (LCEL) provides a declarative way to compose chains using the pipe operator."
    },
    {
        "question": "What do output parsers do in LangChain?",
        "answer": "Output parsers transform the raw text output of an LLM into structured formats such as JSON, lists, or Pydantic models."
    },
    {
        "question": "How does memory work in LangChain?",
        "answer": "Memory components allow conversational applications to retain context across multiple turns of dialogue."
    },
]

# ── Custom evaluators ────────────────────────

def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Asks the LLM to judge whether the RAG answer is correct
    relative to the reference answer on a scale of 1-5.
    """
    reference = example.outputs.get("answer", "")
    prediction = run.outputs.get("output", "")

    judge_prompt = f"""You are an expert evaluator. Score the following answer on a scale of 1 to 5
where 1 = completely wrong and 5 = fully correct and complete.
Respond with ONLY a single integer (1, 2, 3, 4, or 5).

Reference answer: {reference}
Model answer: {prediction}"""

    score_str = llm.invoke(judge_prompt).content.strip()
    try:
        raw = int(score_str[0])  # 1–5
        score = (raw - 1) / 4.0  # normalize to 0.0–1.0
    except (ValueError, IndexError):
        score = 0.0

    return {"key": "correctness", "score": score}


def relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Checks whether the model's answer is relevant to the question (binary).
    """
    question = example.inputs.get("question", "")
    prediction = run.outputs.get("output", "")

    judge_prompt = f"""Does the following answer relevantly address the question?
Question: {question}
Answer: {prediction}

Reply with ONLY 'yes' or 'no'."""

    result = llm.invoke(judge_prompt).content.strip().lower()
    score = 1 if result.startswith("yes") else 0
    return {"key": "relevance", "score": score}


def run_evaluation():
    client = Client()
    dataset_name = "rag-test-dataset"

    # Create (or reuse) a LangSmith dataset
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"question": q["question"]} for q in EVAL_DATASET],
            outputs=[{"answer": q["answer"]} for q in EVAL_DATASET],
            dataset_id=dataset.id,
        )
        print(f"Created dataset '{dataset_name}' with {len(EVAL_DATASET)} examples.")
    else:
        print(f"Reusing existing dataset '{dataset_name}'.")

    # Wrapper so evaluate() can call our RAG chain
    def predict(inputs: dict) -> dict:
        question = inputs["question"]
        output = run_rag(question)
        return {"output": output}

    # Run evaluation
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[correctness_evaluator, relevance_evaluator],
        experiment_prefix="rag-llama-eval",
        metadata={"model": "llama-3.1-70b-instruct", "embedding": "sfr-embedding-mistral"},
    )

    print("\n=== Evaluation complete ===")
    print(f"Results summary: {results}")
    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test
    print("=== Quick RAG test ===")
    response = run_rag("What is LangChain?")
    print(response)

    # Run LangSmith evaluation
    print("\n=== Running LangSmith evaluation ===")
    run_evaluation()