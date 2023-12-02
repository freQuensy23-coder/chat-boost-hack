import os
from glob import glob

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma


def format_query(query: str) -> str:
    return 'query: {0}'.format(query)


def format_passage(passage: str) -> str:
    return 'passage: {0}'.format(passage)


def get_text_splitter(
    separator: str, chunk_size: int, chunk_overlap: int,
) -> CharacterTextSplitter:
    return CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_embedder(device: str = 'cpu') -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-small',
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'normalize_embeddings': True},
    )


def get_vector_store() -> Chroma:
    embedder = get_embedder()
    return Chroma(embedding_function=embedder)


def get_retriever(documents_dir: str) -> ParentDocumentRetriever:
    child_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=10, chunk_overlap=0,
    )
    parent_splitter = CharacterTextSplitter(
        separator='\n\n', chunk_size=10, chunk_overlap=0,
    )
    retriever = ParentDocumentRetriever(
        vectorstore=get_vector_store(),
        docstore=InMemoryStore(),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(load_documents(documents_dir))
    return retriever


def load_documents(documents_dir: str) -> list[Document]:
    document_paths = glob(
        os.path.join(documents_dir, '**/*.txt'), recursive=True,
    )
    documents = []
    for document_path in document_paths:
        with open(document_path) as document_file:
            document_content = document_file.read().strip()
            documents.append(
                Document(page_content=document_content, source=document_path),
            )
    return documents


if __name__ == '__main__':
    retriever = get_retriever(documents_dir='documents')
    relevant_documents = retriever.get_relevant_documents(
        format_query('Я хотел бы настроить переадресацию'),
    )
