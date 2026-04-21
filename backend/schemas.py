from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from PIL import Image


class SimplifyTextRequest(BaseModel):
    text: str = Field(..., description="The text to be simplified.")
    custom_context: str | None = Field(
        None, description="Optional custom context to aid in simplification."
    )
    unalterable_terms_text: str | None = Field(
        None, description="Optional comma-separated terms that should not be altered during simplification."
    )
    target_language: str | None = Field(
        None, description="Optional target language for translation (e.g., 'Swahili', 'French')."
    )

class SimplifiedSentence(BaseModel):
    sentence: str
    image_prompt: str
    translated_sentence: Optional[str] = None

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
    translated_sentence: Optional[str] = None

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
    symbolset: str = Field(default="arasaac", description="Symbol library to use (e.g., 'arasaac', 'mulberry', 'sclera')")
    use_global_symbols: bool = Field(default=True, description="Whether to use Global Symbols API (fallback to local generation if False or not found)")


class GenerateIconsResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for the icon generation request.")
    icons: List[GeneratedIcon] = Field(..., description="List of generated icons corresponding to the revised sentences.")


# Separated symbol search + AI generation flow
class SymbolSearchRequest(BaseModel):
    sentences: List[RevisedSentence]
    symbolset: str = Field(default="arasaac")


class SymbolSearchResult(RevisedSentence):
    symbol_found: bool
    symbol_image_path: Optional[str] = None


class SymbolSearchResponse(BaseModel):
    request_id: str
    results: List[SymbolSearchResult]


class AIGenerateSentence(BaseModel):
    sentence: str
    ai_prompt: str
    highlighted: bool
    symbol_image_path: Optional[str] = None


class AIGenerateRequest(BaseModel):
    request_id: str
    sentences: List[AIGenerateSentence]


# TTS schemas
class TTSSentence(BaseModel):
    id: int
    text: str


class TTSSynthesizeRequest(BaseModel):
    sentences: List[TTSSentence]
    request_id: Optional[str] = None
    voice: str = "af_heart"
    speed: float = 1.0


class TTSAudioFile(BaseModel):
    id: int
    filename: str


class TTSSynthesizeResponse(BaseModel):
    request_id: str
    audio_files: List[TTSAudioFile]