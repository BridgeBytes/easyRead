from pydantic import BaseModel, Field, field_validator
from enum import Enum

class SymbolLibrary(str, Enum):
    OPEN_MOJI = "openmoji"
    ARASAAC = "arasaac"
    LDS = "lds"


class SimplifyTextRequest(BaseModel):
    text: str = Field(..., description="The text to be simplified.")
    custom_context: str | None = Field(
        "custom_context", description="Optional custom context to aid in simplification."
    )
    unalterable_terms_text: str | None = Field(
        "unalterable_terms_text", description="Optional comma-separated terms that should not be altered during simplification."
    )
    num_of_images: int = Field(default=1, description="Number of images to generate for the simplified text.")
    symbol_library: SymbolLibrary = Field(
        SymbolLibrary.OPEN_MOJI, description="The symbol library to use for generating icons."
    )

    @property
    def sentences(self) -> list[str]:
        return [s.strip() for s in self.text.split('.') if s.strip()]


class Sentences(BaseModel):
    sentence: str = Field(description="The simplified sentence.")
    image_prompt: str = Field(description="The prompt used for generating the image/icon.")


class SimplifyTextResponse(BaseModel):
    title: str | None = Field(description="The title of the simplified text.")
    sentences: list[Sentences] | None = Field(description="List of simplified sentences with image prompts.")


class GenerateIconRequest(BaseModel):
    pass


class GenerateIconsResponse(BaseModel):
    pass