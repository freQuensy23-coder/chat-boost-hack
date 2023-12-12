from fastapi import APIRouter, Request, BackgroundTasks

from dependencies import SuccessAnalyzer
from schemas import Dialogue

router = APIRouter()


@router.post('/completion')
def completion(
    dialogue: Dialogue, request: Request, background_tasks: BackgroundTasks,
) -> str:
    query = dialogue.get_query()
    relevant_documents = []
    if query:
        relevant_documents = request.state.retriever.get_relevant_documents(query)

    response = request.state.llm(dialogue.format_dialogue(), relevant_documents)

    background_tasks.add_task(
        log_metrics, dialogue, request.state.success_analyzer,
    )

    return response


def log_metrics(dialogue: Dialogue, success_analyzer: SuccessAnalyzer) -> None:
    success_analyzer(dialogue.format_dialogue())
