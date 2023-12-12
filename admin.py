import uuid

from fastapi import APIRouter

from schemas import AdminTurn, SuccessMetric

router = APIRouter()


@router.get('/dialogues/{dialogue_id}')
def get_dialogue(dialogue_id: uuid.UUID) -> list[AdminTurn]:
    return []


@router.get('/metrics/success_metric')
def get_success_metric() -> SuccessMetric:
    return SuccessMetric(n_started_dialogues=10, n_succeeded_dialogues=3)
