import os
from glob import glob

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS


def format_query(query: str) -> str:
    return 'query: {0}'.format(query)


def format_passage(passage: str) -> str:
    return 'passage: {0}'.format(passage)


def get_embedder(device: str = 'cpu') -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-small',
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'normalize_embeddings': True},
    )


def get_vector_store(
    documents_dir: str, vector_store_dir: str, use_cache: bool = False,
) -> FAISS:
    embedder = get_embedder()
    documents = load_documents(documents_dir)

    if use_cache and os.path.isdir(vector_store_dir):
        vector_store = FAISS.load_local(vector_store_dir, embedder)
    else:
        vector_store = FAISS.from_documents(documents, embedder)

    if use_cache:
        vector_store.save_local(vector_store_dir)

    return vector_store


def load_documents(documents_dir: str) -> list[Document]:
    documents = []
    document_paths = glob(
        os.path.join(documents_dir, '**/*.txt'), recursive=True,
    )
    for document_path in document_paths:
        with open(document_path) as document_file:
            document_content = document_file.read().strip()
        for document_chunk in split_document(document_content):
            documents.append(
                Document(
                    page_content=document_chunk, source=document_path,
                ),
            )
    return documents


def split_document(document_content: str) -> list[str]:
    chunks = [
        *document_content.split('\n'),
        *document_content.split('\n\n'),
        document_content,
    ]
    return [format_passage(chunk) for chunk in chunks]


if __name__ == '__main__':
    vector_store = get_vector_store(
        documents_dir='documents',
        vector_store_dir='vector_store',
    )
    relevant_documents = vector_store.similarity_search_with_relevance_scores(
        format_query('Что такое бонусы "спасибо"'),
    )
