from fastapi import APIRouter
from pydantic import BaseModel
from therapygeneration.services.word_engine import PracticeService

router = APIRouter()
service = PracticeService()


class PracticeRequest(BaseModel):
    child_id: str
    letter: str
    mode: str          # starts_with | contains | ends_with
    level: int
    count: int


@router.post("/practice")
def create_practice(req: PracticeRequest):
    return service.create_activity(
        child_id=req.child_id,
        letter=req.letter,
        mode=req.mode,
        level=req.level,
        count=req.count
    )
