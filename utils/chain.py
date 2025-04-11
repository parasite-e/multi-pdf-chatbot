#
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def answer_question(vectorstore, query):
    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
        token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
        task='text-generation',
        temperature=0.6,  # Reduce variability
        max_new_tokens=512  # Ensure sufficient output length
    )

    # Define retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Define prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant. Using only the following context, provide a concise and accurate answer to the question. If the context lacks sufficient information, summarize what is relevant without saying "I don't know" or similar phrases.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke chain
    answer = rag_chain.invoke(query)
    return answer
