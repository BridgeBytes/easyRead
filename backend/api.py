from logging import getLogger
from fastapi import FastAPI
from schemas import SimplifyTextRequest, SimplifiedTextResponse, GenerateIconRequest ,GenerateIconsResponse
from controller import Controller

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
    pass
    