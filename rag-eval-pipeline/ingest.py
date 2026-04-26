"""
Run this script once to ingest sample documents into the vector store.
Usage: python ingest.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector_store import ingest_documents


def main():
    print("Ingesting HR policy documents...")
    hr_chunks = ingest_documents(
        docs_dir="data/sample_docs",
        collection_name="hr_policies",
    )
    print(f"HR policies: {hr_chunks} chunks ingested")

    print("Ingesting care plan documents...")
    care_chunks = ingest_documents(
        docs_dir="data/sample_docs",
        collection_name="care_plans",
    )
    print(f"Care plans: {care_chunks} chunks ingested")

    print("Done. Vector store is ready.")


if __name__ == "__main__":
    main()
