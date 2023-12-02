import math
import os
from glob import glob
from typing import Literal

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class WeightedParentDocumentRetriever(ParentDocumentRetriever):
    aggregation: Literal['min', 'mean'] = 'min'
    length_normalization: bool = False

    def _get_total_score(self, scores: list[float]) -> float:
        if self.aggregation == 'min':
            return min(scores)
        if self.aggregation == 'mean':
            return sum(scores) / len(scores)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
    ) -> list[tuple[Document, float]]:
        child_documents = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs,
        )

        document_ids = {}
        for child_document in child_documents:
            document_id = child_document[0].metadata[self.id_key]
            if document_id not in document_ids:
                document_ids[document_id] = {'scores': []}
            score = child_document[1]
            if self.length_normalization:
                score /= math.log(len(child_document[0].page_content))
            document_ids[document_id]['scores'].append(score)

        for document_id in document_ids:
            total_score = self._get_total_score(
                document_ids[document_id]['scores'],
            )
            document_ids[document_id]['total_score'] = total_score

        documents = self.docstore.mget(list(document_ids.keys()))

        total_scores = [
            document_ids[document_id]['total_score']
            for document_id in document_ids
        ]

        scored_documents = []
        for idx, document in enumerate(documents):
            if document is None:
                continue
            scored_documents.append((document, total_scores[idx]))

        return sorted(scored_documents, key=lambda document: document[1])


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


def get_retriever(documents_dir: str) -> WeightedParentDocumentRetriever:
    child_splitter = CharacterTextSplitter(
        separator='\n', chunk_size=10, chunk_overlap=0,
    )
    parent_splitter = CharacterTextSplitter(
        separator='\n\n', chunk_size=10, chunk_overlap=0,
    )
    retriever = WeightedParentDocumentRetriever(
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
    relevant_documents: list[tuple[Document, float]] = retriever.get_relevant_documents(
        format_query('Я хотел бы настроить переадресацию'),
        search_kwargs={'k': 10}, # control the number of retrieved child documents
    )
