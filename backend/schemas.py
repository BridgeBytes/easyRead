from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from PIL import Image


class SimplifyTextRequest(BaseModel):
    text: str = Field(..., description="The text to be simplified.")
    custom_context: str | None = Field(
        "custom_context", description="Optional custom context to aid in simplification."
    )
    unalterable_terms_text: str | None = Field(
        "unalterable_terms_text", description="Optional comma-separated terms that should not be altered during simplification."
    )

class SimplifiedSentence(BaseModel):
    sentence: str
    image_prompt: str

class SimplifiedText(BaseModel):
    title: str
    simplified_sentences: List[SimplifiedSentence]

class Validation(BaseModel):
    missing_info: str
    extra_info: str
    other_feedback: str

class RevisedSentence(BaseModel):
    sentence: str
    image_prompt: str
    highlighted: bool

class Revision(BaseModel):
    revised_sentences: List[RevisedSentence]

class SimplifiedTextResponse(BaseModel):
    simplified_text: SimplifiedText
    validation: Validation
    revision: Revision

class SymbolLibrary(str, Enum):
    OPEN_MOJI = "openmoji"
    ARASAAC = "arasaac"
    LDS = "lds"


class GeneratedIcon(RevisedSentence):
    image_path: str = Field(..., description="Path to the generated icon image.")


class GenerateIconRequest(BaseModel):
    sentences: List[RevisedSentence] = Field(..., description="List of revised sentences for which to generate icons.")


class GenerateIconsResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the icon generation request.")
    icons: List[GeneratedIcon] = Field(..., description="List of generated icons corresponding to the revised sentences.")