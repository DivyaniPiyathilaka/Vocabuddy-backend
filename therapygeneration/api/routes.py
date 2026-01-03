from fastapi import APIRouter

router = APIRouter(prefix="/therapy", tags=["therapy"])

@router.get("/health")
def health():
    return {"ok": True}
