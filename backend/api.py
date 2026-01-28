from logging import getLogger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from schemas import SimplifyTextRequest, SimplifiedTextResponse, GenerateIconRequest ,GenerateIconsResponse
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
async def generate_icons(sentences: GenerateIconRequest) -> GenerateIconsResponse:
    sentences = sentences.model_dump()['sentences']
    response = controller.generate_icons(sentences)
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