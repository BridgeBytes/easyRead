import io
import os
from logging import getLogger
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from schemas import (
    SimplifyTextRequest, SimplifiedTextResponse,
    GenerateIconRequest, GenerateIconsResponse,
    SymbolSearchRequest, SymbolSearchResponse,
    AIGenerateRequest,
    TTSSynthesizeRequest, TTSSynthesizeResponse,
)
from controller import Controller
from services.config import Config

TTS_URL = os.getenv("TTS_URL", "http://tts.easyread:8001")

logger = getLogger(__name__)

api = FastAPI(
    title="EasyRead Backend API",
    version="0.1.0"
)

controller = Controller()

@api.post("/sentence/simplify", tags=["Sentence"])
async def simplify_sentence(text: SimplifyTextRequest) -> SimplifiedTextResponse:
    target_lang = text.target_language
    simple_text: dict = controller.simplify_text(text.text, target_language=target_lang)
    validation = controller.validate_text(text.text, simple_text['simplified_sentences'])
    revision = controller.revise_text(text.text, simple_text['simplified_sentences'], validation, target_language=target_lang)
    return SimplifiedTextResponse(simplified_text=simple_text,validation=validation,revision=revision)

@api.post("/sentence/translate", tags=["Sentence"])
async def translate_text(text: str, target_language: str) -> str:
    return controller.translate_text(text, target_language)


@api.post("/sentence/generate-icons", tags=["Sentence"])
async def generate_icons(request: GenerateIconRequest) -> GenerateIconsResponse:
    request_data = request.model_dump()
    response = controller.generate_icons(
        sentences=request_data['sentences'],
        symbolset=request_data.get('symbolset', 'arasaac'),
        use_global_symbols=request_data.get('use_global_symbols', True)
    )
    return GenerateIconsResponse(request_id=response["request_id"], icons=response["icons"])


@api.post("/sentence/search-symbols", tags=["Sentence"])
async def search_symbols(request: SymbolSearchRequest) -> SymbolSearchResponse:
    request_data = request.model_dump()
    response = controller.search_symbols(
        sentences=request_data['sentences'],
        symbolset=request_data.get('symbolset', 'arasaac'),
    )
    return SymbolSearchResponse(request_id=response["request_id"], results=response["results"])


@api.post("/sentence/generate-ai-icons", tags=["Sentence"])
async def generate_ai_icons(request: AIGenerateRequest) -> GenerateIconsResponse:
    request_data = request.model_dump()
    response = controller.generate_ai_icons(
        request_id=request_data['request_id'],
        sentences=request_data['sentences'],
    )
    return GenerateIconsResponse(request_id=response["request_id"], icons=response["icons"])


@api.get("/icons/{request_id}/{image_id}", tags=["Icons"])
async def get_icons(request_id: str, image_id: str):
    config = Config()
    image_path = config.ICON_OUTPUT_PATH / request_id / image_id

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Icon not found")

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=image_id
    )


@api.post("/sentence/synthesize", tags=["Audio"])
async def synthesize_audio(request: TTSSynthesizeRequest) -> TTSSynthesizeResponse:
    """Proxy TTS synthesis request to the TTS service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TTS_URL}/synthesize",
            json=request.model_dump(),
            timeout=120,
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="TTS service error")
        return TTSSynthesizeResponse(**response.json())


@api.get("/audio/{request_id}/file/{filename}", tags=["Audio"])
async def get_audio_file(request_id: str, filename: str):
    """Proxy a single WAV audio file from the TTS service."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TTS_URL}/audio/{request_id}/file/{filename}",
            timeout=30,
        )
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Audio file not found")
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


@api.get("/audio/{request_id}/export", tags=["Audio"])
async def export_audio_zip(request_id: str):
    """Proxy ZIP export of all audio files from the TTS service."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TTS_URL}/audio/{request_id}/export",
            timeout=60,
        )
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Audio not found")
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="easyread_audio.zip"'
            },
        )