from langchain_community.vectorstores import FAISS
from utils.embedder import get_embeddings


def create_vectorstore(docs):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )
    return vectorstore
