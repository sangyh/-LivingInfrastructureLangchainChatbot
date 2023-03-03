"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from llama_index import download_loader

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(Path("../../config/.env"))

def fetchNotiondocs():
    NotionPageReader = download_loader('NotionPageReader')
    integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
    reader = NotionPageReader(integration_token=integration_token)

    page_ids = reader.query_database("e3624d062dc64142b33d6ef797c0fe24")
    documents = reader.load_data(page_ids=page_ids)
    LCdocs = []
    for doc in documents:
        LCdocs.append(doc.to_langchain_format())
    
    return LCdocs


def ingest_docs():
    """Get documents from web pages."""
    #loader = ReadTheDocsLoader("langchain.re3adthedocs.io/en/latest/")
    raw_documents = fetchNotiondocs() #loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
