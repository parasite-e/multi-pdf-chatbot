from langchain_community.document_loaders import UnstructuredFileLoader, WebBaseLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import tempfile
import logging
from io import BytesIO

# Suppress CropBox warning
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def load_and_split(input_type, input_data):
    docs = []

    # Handle different input types
    if input_type == 'Text':
        # Expect input_data as a string
        if not input_data.strip():
            raise ValueError("Text input cannot be empty")
        docs = [Document(page_content=input_data)]

    elif input_type == 'Link':
        # Expect input_data as a list of URLs
        if not input_data:
            raise ValueError("No URLs provided")
        # WebBaseLoader accepts a list of URLs
        loader = WebBaseLoader(input_data)
        docs = loader.load()

    else:
        # Handle file-like inputs (PDF, DOCX, TXT)
        if not input_data:
            raise ValueError("No file provided")
        with tempfile.NamedTemporaryFile(delete=False, suffix=getattr(input_data, 'name', '.tmp')) as tmp:
            tmp.write(input_data.read())
            temp_path = tmp.name

        if input_type == 'PDF':
            loader = UnstructuredFileLoader(temp_path)
            docs = loader.load()
        elif input_type == 'DOCX':
            loader = UnstructuredWordDocumentLoader(temp_path)
            docs = loader.load()
        elif input_type == 'TXT':
            loader = TextLoader(temp_path)
            docs = loader.load()
        else:
            raise ValueError('Unsupported input type')

        os.remove(temp_path)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    split_docs = text_splitter.split_documents(docs)

    # Filter out short or empty chunks
    filtered_docs = [doc for doc in split_docs if len(
        doc.page_content.strip()) > 50]

    if not filtered_docs:
        raise ValueError("No valid document chunks after processing")

    return filtered_docs
