from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI

import admin
import chat
from dependencies import LLM, Retriever, SuccessAnalyzer


@asynccontextmanager
async def load_resources(app: FastAPI) -> AsyncGenerator[dict[str, Any], None]:
    # pylint: disable=unused-argument
    yield {
        'retriever': Retriever(),
        'llm': LLM(),
        'success_analyzer': SuccessAnalyzer(),
    }


app = FastAPI(lifespan=load_resources)
app.include_router(chat.router, tags=['chat'], prefix='/chat')
app.include_router(admin.router, tags=['admin'], prefix='/admin')
