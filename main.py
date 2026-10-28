
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, data
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
embedding_model=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore=Chroma(
    collection_name="deep_learning",
    persist_directory="chroma_db_rate_limited",
    embedding_function=embedding_model
    )
print("Total stored chunks:", vectorstore._collection.count())
retriever=vectorstore.as_retriever(
search_type='mmr',
search_kwargs={
    "k":4,
    'fetch_k':10,
    'lambda_mult':0.5
}
)
llm=ChatGoogleGenerativeAI(model="gemini-3.8-flash")
prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant that answers questions based on the context provided.IF answer is not there write i could'nt find it"),
    ("human", """Context:
    {context}
    
    Question:
    {question}
    """)
])
print("RAG system created")
print("press 0 to exit")
while True:
    query=input("you:")
    if query=="0":
        break
    # docs=retriever.invoke(query)
    docs = vectorstore.similarity_search(query, k=5)

    print("\n--- RETRIEVED CHUNKS ---")
    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i}")
        print(doc.page_content[:1000])
        print("Metadata:", doc.metadata)
        print("-" * 50)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt=prompt.invoke(
    {"context":context,
    "question":query}
    )
    response=llm.invoke(final_prompt)
    print(f"\n AI:{response.text}")