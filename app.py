from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request

from dependencies import LLM, Retriever, SuccessAnalyzer
from schemas import Actor, Dialogue, Turn


@asynccontextmanager
async def load_resources(app: FastAPI) -> AsyncGenerator[dict[str, Any], None]:
    # pylint: disable=unused-argument
    yield {
        'retriever': Retriever(),
        'llm': LLM(),
        'success_analyzer': SuccessAnalyzer(),
    }


app = FastAPI(lifespan=load_resources)


@app.post('/completion')
def completion(
    dialogue: Dialogue, request: Request, background_tasks: BackgroundTasks,
) -> Dialogue:
    query = dialogue.get_query()
    relevant_documents = []
    if query:
        relevant_documents = request.state.retriever.get_relevant_documents(query)

    response = request.state.llm(dialogue.format_dialogue(), relevant_documents)

    background_tasks.add_task(
        log_metrics, dialogue, request.state.success_analyzer,
    )

    dialogue.turns.append(Turn(actor=Actor.ASSISTANT.value, utterance=response))

    return dialogue


def log_metrics(dialogue: Dialogue, success_analyzer: SuccessAnalyzer) -> None:
    success_analyzer(dialogue.format_dialogue())
