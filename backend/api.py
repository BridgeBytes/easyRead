from logging import getLogger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from schemas import (
    SimplifyTextRequest, SimplifiedTextResponse,
    GenerateIconRequest, GenerateIconsResponse,
    SymbolSearchRequest, SymbolSearchResponse,
    AIGenerateRequest,
)
from controller import Controller
from services.config import Config

logger = getLogger(__name__)

api = FastAPI(
    title="EasyRead Backend API",
    version="0.1.0"
)

controller = Controller()

@api.post("/sentence/simplify", tags=["Sentence"])
async def simplify_sentence(text: SimplifyTextRequest) -> SimplifiedTextResponse:
    simple_text: dict = controller.simplify_text(text.text)
    validation = controller.validate_text(text.text, simple_text['simplified_sentences'])
    revision = controller.revise_text(text.text, simple_text['simplified_sentences'], validation)
    return SimplifiedTextResponse(simplified_text=simple_text,validation=validation,revision=revision)


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