from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import time
load_dotenv()
data=PyPDFLoader("document loaders/deep-learning.pdf")
docs = data.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks=splitter.split_documents(docs)
embedding_model=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore=Chroma(
    collection_name="deep_learning",
    embedding_function=embedding_model,
    persist_directory="chroma_db_rate_limited"
)
batch_size = 10

for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]

    vectorstore.add_documents(batch)

    print(f"Stored {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

    # Pause between batches to reduce request pressure
    if start + batch_size < len(chunks):
        time.sleep(2)

print("All chunks stored successfully!")