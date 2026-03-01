from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langsmith import traceable
# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.environ["NAVIGATOR-API-PROJECTS"]

# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────
docs = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
    Document(page_content="It enables applications that use a language model as reasoning element.", metadata={"source": "doc2"}),
    Document(page_content="RAG combines retrieval of relevant documents with generation using LLMs.", metadata={"source": "doc3"}),
]

# ─────────────────────────────────────────────
# Text splitting
# ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"Original docs: {len(docs)}, Split chunks: {len(splits)}")

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
print("Embedding model initialized.")

# ─────────────────────────────────────────────
# Vector store + retriever
# ─────────────────────────────────────────────
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever initialized.")

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
# Prompt
# ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}"""
)

# ─────────────────────────────────────────────
# STATE — the shared "memory" passed between nodes
# ─────────────────────────────────────────────
# In LangGraph, every node reads from and writes to this State object.
# Think of it as a whiteboard that every step in the graph can see and update.
class RAGState(TypedDict):
    question: str           # The user's question (set at the start)
    context: str            # Retrieved docs as plain text (set by retrieve node)
    answer: str             # Final generated answer (set by generate node)

# ─────────────────────────────────────────────
# NODE 1: Retrieve
# ─────────────────────────────────────────────
# This node takes the question from state, searches the vector store,
# formats the results into plain text, and saves it back to state as "context".
def retrieve(state: RAGState) -> RAGState:
    print("--- NODE: retrieve ---")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return {"context": context}  # only update the "context" field in state

# ─────────────────────────────────────────────
# NODE 2: Generate
# ─────────────────────────────────────────────
# This node reads both "question" and "context" from state,
# fills them into the prompt, sends it to the LLM, and saves the answer to state.
def generate(state: RAGState) -> RAGState:
    print("--- NODE: generate ---")
    filled_prompt = prompt.invoke({
        "question": state["question"],
        "context": state["context"]
    })
    response = llm.invoke(filled_prompt)
    answer = StrOutputParser().invoke(response)
    return {"answer": answer}  # only update the "answer" field in state

# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────
# 1. Create a graph that uses RAGState as its shared state
# 2. Add nodes (the steps/functions)
# 3. Add edges (the connections between steps)
# 4. Set entry point (where the graph starts)
# 5. Compile it into a runnable app

graph = StateGraph(RAGState)

# Add nodes — each node has a name and the function it runs
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

# Add edges — defines the order: retrieve → generate → END
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

# Set the entry point — where the graph starts
graph.set_entry_point("retrieve")

# Compile — turns the graph definition into something you can invoke
app = graph.compile()

# ─────────────────────────────────────────────
# RUN THE GRAPH — wrapped with LangSmith tracing
# ─────────────────────────────────────────────
# @traceable records every call to run_rag in the LangSmith dashboard.
# It logs the input question, output answer, timing, and all internal steps.
@traceable(name="rag_langgraph_pipeline")
def run_rag(question: str) -> str:
    initial_state = {"question": question}
    result = app.invoke(initial_state)
    return result["answer"]

# Quick smoke test
result = app.invoke({"question": "What is LangChain?"})

print("\n=== Final Answer ===")
print(result["answer"])

# You can also inspect the full state to see everything
print("\n=== Full State ===")
print(f"Question : {result['question']}")
print(f"Context  : {result['context']}")
print(f"Answer   : {result['answer']}")

# ─────────────────────────────────────────────
# LANGSMITH EVALUATION
# ─────────────────────────────────────────────

# 10 test questions with reference (ground-truth) answers
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

# ── Evaluator 1: Correctness ─────────────────
# Asks the LLM to score the answer 1-5 vs the reference answer.
# Normalized to 0.0-1.0 because LangSmith requires scores in that range.
def correctness_evaluator(run: Run, example: Example) -> dict:
    reference = example.outputs.get("answer", "")
    prediction = run.outputs.get("output", "")

    judge_prompt = f"""You are an expert evaluator. Score the following answer on a scale of 1 to 5
where 1 = completely wrong and 5 = fully correct and complete.
Respond with ONLY a single integer (1, 2, 3, 4, or 5).

Reference answer: {reference}
Model answer: {prediction}"""

    score_str = llm.invoke(judge_prompt).content.strip()
    try:
        raw = int(score_str[0])       # get the integer 1-5
        score = (raw - 1) / 4.0       # normalize: 1→0.0, 3→0.5, 5→1.0
    except (ValueError, IndexError):
        score = 0.0

    return {"key": "correctness", "score": score}


# ── Evaluator 2: Relevance ───────────────────
# Asks the LLM a simple yes/no — did the answer actually address the question?
# Returns 1 for yes, 0 for no.
def relevance_evaluator(run: Run, example: Example) -> dict:
    question = example.inputs.get("question", "")
    prediction = run.outputs.get("output", "")

    judge_prompt = f"""Does the following answer relevantly address the question?
Question: {question}
Answer: {prediction}

Reply with ONLY 'yes' or 'no'."""

    result = llm.invoke(judge_prompt).content.strip().lower()
    score = 1 if result.startswith("yes") else 0
    return {"key": "relevance", "score": score}


# ── run_evaluation ───────────────────────────
def run_evaluation():
    client = Client()
    dataset_name = "rag-langgraph-dataset"

    # Create the dataset in LangSmith only if it doesn't already exist
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

    # predict() is the bridge between LangSmith and our LangGraph app.
    # LangSmith calls predict({"question": "..."}) for each example
    # and expects back {"output": "..."} — so we wrap run_rag() to match that format.
    def predict(inputs: dict) -> dict:
        question = inputs["question"]
        output = run_rag(question)
        return {"output": output}

    # Run the full evaluation:
    # - calls predict() for all 10 questions
    # - passes each result through both evaluators
    # - uploads scores and traces to LangSmith dashboard
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[correctness_evaluator, relevance_evaluator],
        experiment_prefix="rag-langgraph-eval",
        metadata={
            "model": "llama-3.1-70b-instruct",
            "embedding": "sfr-embedding-mistral",
            "framework": "langgraph"        # extra label so you can tell this apart from the LangChain version
        },
    )

    print("\n=== Evaluation complete ===")
    print(f"Results summary: {results}")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Running LangSmith Evaluation ===")
    run_evaluation()