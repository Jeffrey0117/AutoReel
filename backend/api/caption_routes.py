"""
Caption API Routes - IG caption generation, example management, draft listing.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.caption_service import caption_service

router = APIRouter(prefix="/api/caption", tags=["caption"])


class ExampleAction(BaseModel):
    action: str  # "add" or "delete"
    text: Optional[str] = None
    index: Optional[int] = None


class GenerateRequest(BaseModel):
    draft_name: str


class BatchGenerateRequest(BaseModel):
    draft_names: List[str]


class CaptionUpdate(BaseModel):
    caption: str


# --- Examples ---

@router.get("/examples")
async def get_examples():
    return {"examples": caption_service.get_examples()}


@router.post("/examples")
async def manage_examples(body: ExampleAction):
    if body.action == "add":
        if not body.text or not body.text.strip():
            raise HTTPException(400, "text is required for add action")
        examples = caption_service.add_example(body.text.strip())
        return {"success": True, "examples": examples}

    if body.action == "delete":
        if body.index is None:
            raise HTTPException(400, "index is required for delete action")
        examples = caption_service.delete_example(body.index)
        return {"success": True, "examples": examples}

    raise HTTPException(400, f"Unknown action: {body.action}")


# --- Drafts ---

@router.get("/drafts")
async def list_drafts():
    return {"drafts": caption_service.list_drafts()}


# --- Captions ---

@router.get("/captions")
async def list_captions():
    return {"captions": caption_service.list_captions()}


@router.post("/generate")
async def generate_caption(body: GenerateRequest):
    if caption_service.is_generating:
        return {"success": False, "error": "正在批次生成中，請稍候"}
    result = caption_service.generate_caption(body.draft_name)
    return result


@router.post("/generate-batch")
async def generate_batch(body: BatchGenerateRequest):
    if not body.draft_names:
        raise HTTPException(400, "draft_names is required")

    started = caption_service.start_batch_generation(body.draft_names)
    if not started:
        return {"success": False, "error": "已在生成中"}
    return {"success": True, "message": f"開始批次生成 {len(body.draft_names)} 篇文案"}


@router.patch("/captions/{name}/publish")
async def toggle_publish(name: str):
    is_published = caption_service.toggle_published(name)
    return {"success": True, "published": is_published}


@router.put("/captions/{name}")
async def update_caption(name: str, body: CaptionUpdate):
    updated = caption_service.update_caption(name, body.caption)
    if not updated:
        raise HTTPException(404, f"Caption not found: {name}")
    return {"success": True}


@router.delete("/captions/{name}")
async def delete_caption(name: str):
    deleted = caption_service.delete_caption(name)
    if not deleted:
        raise HTTPException(404, f"Caption not found: {name}")
    return {"success": True}
