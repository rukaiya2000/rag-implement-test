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
# Load environment variables from .ENV file
load_dotenv()

api_key = os.environ["NAVIGATOR-API-PROJECTS"]

docs = [
    Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
    Document(page_content="It enables applications that use a language model as reasoning element.", metadata={"source": "doc2"}),
    Document(page_content="RAG combines retrieval of relevant documents with generation using LLMs.", metadata={"source": "doc3"}),
]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Chunk size in characters
    chunk_overlap=50    # Overlap for context preservation
)
splits = text_splitter.split_documents(docs)

print(f"Original docs: {len(docs)}, Split chunks: {len(splits)}")
print("Sample split chunk:", splits[0].page_content[:200], "...")

# Embeddings with sfr-embedding-mistral (Mistral AI model)
embeddings = OpenAIEmbeddings(
    model="sfr-embedding-mistral" ,
    openai_api_base="https://api.ai.it.ufl.edu",
    api_key=api_key ,
    tiktoken_enabled=False,
    check_embedding_ctx_length=False
 
)
print("Embedding model initialized.")

# Create Chroma vector store (in-memory, persists to ./chroma_db)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Optional: save locally
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 docs

# OpenAI LLM for generation
llm = ChatOpenAI(
    openai_api_base="https://api.ai.it.ufl.edu",
    model = "llama-3.1-70b-instruct",
    api_key=api_key,
    temperature=0.1
)

# RAG prompt template
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}"""
)

# RAG chain: retrieve -> format docs -> prompt -> LLM -> parse
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query the RAG system
response = rag_chain.invoke("What is LangChain?")
print(response)