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
# RUN THE GRAPH
# ─────────────────────────────────────────────
# We pass in the initial state with just the question.
# The graph fills in "context" and "answer" as it runs through nodes.
initial_state = {"question": "What is LangChain?"}
result = app.invoke(initial_state)

print("\n=== Final Answer ===")
print(result["answer"])

# You can also inspect the full state to see everything
print("\n=== Full State ===")
print(f"Question : {result['question']}")
print(f"Context  : {result['context']}")
print(f"Answer   : {result['answer']}")