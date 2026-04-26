from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from src.config import get_settings
from typing import List, Optional
import os

settings = get_settings()


def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_vector_store(collection_name: str):
    embeddings = get_embeddings()

    if settings.vector_store == "milvus":
        from langchain_milvus import Milvus
        uri = settings.zilliz_uri or f"http://{settings.milvus_host}:{settings.milvus_port}"
        return Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args={"uri": uri, "token": settings.zilliz_token},
        )

    # Default: Chroma (local, no extra setup needed)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=os.path.join(settings.chroma_persist_dir, collection_name),
    )


def ingest_documents(docs_dir: str, collection_name: str) -> int:
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    store = get_vector_store(collection_name)
    store.add_documents(chunks)

    return len(chunks)


def retrieve(query: str, collection_name: str, k: int = 4) -> tuple[List[Document], str]:
    store = get_vector_store(collection_name)
    docs = store.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    return docs, context
