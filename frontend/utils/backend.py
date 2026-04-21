"""
Backend API Client for EasyRead

Handles all communication with the EasyRead backend service.
"""

import os
import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class SimplifiedSentence:
    sentence: str
    image_prompt: str
    translated_sentence: Optional[str] = None


@dataclass
class RevisedSentence:
    sentence: str
    image_prompt: str
    highlighted: bool
    translated_sentence: Optional[str] = None


@dataclass
class GeneratedIcon:
    sentence: str
    image_prompt: str
    highlighted: bool
    image_path: str
    translated_sentence: Optional[str] = None


@dataclass
class SimplifyResponse:
    title: str
    simplified_sentences: list[SimplifiedSentence]
    validation: dict
    revised_sentences: list[RevisedSentence]


@dataclass
class IconsResponse:
    request_id: str
    icons: list[GeneratedIcon]


@dataclass
class SymbolSearchResult:
    sentence: str
    image_prompt: str
    highlighted: bool
    symbol_found: bool
    symbol_image_path: Optional[str]


@dataclass
class SymbolSearchResponse:
    request_id: str
    results: list[SymbolSearchResult]


@dataclass
class TTSAudioFile:
    id: int
    filename: str


@dataclass
class TTSResponse:
    request_id: str
    audio_files: list[TTSAudioFile]


class BackendClient:
    """Client for communicating with the EasyRead backend API."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the backend client.

        Args:
            base_url: Backend API base URL. Defaults to BACKEND_URL env var
                      or http://backend.easyread:8000 for Docker environment.
        """
        self.base_url = base_url or os.getenv(
            "BACKEND_URL", "http://backend.easyread:8000"
        )

    def simplify_text(
        self,
        text: str,
        custom_context: Optional[str] = None,
        unalterable_terms: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> SimplifyResponse:
        """
        Send text to the backend for simplification.

        Args:
            text: The text to be simplified
            custom_context: Optional context to aid in simplification
            unalterable_terms: Optional comma-separated terms to preserve
            target_language: Optional target language for translation

        Returns:
            SimplifyResponse with title, simplified sentences, validation, and revised sentences

        Raises:
            requests.RequestException: If the API request fails
        """
        payload = {"text": text}

        if custom_context:
            payload["custom_context"] = custom_context
        if unalterable_terms:
            payload["unalterable_terms_text"] = unalterable_terms
        if target_language:
            payload["target_language"] = target_language

        response = requests.post(
            f"{self.base_url}/sentence/simplify",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        simplified_sentences = [
            SimplifiedSentence(
                sentence=s["sentence"],
                image_prompt=s["image_prompt"],
                translated_sentence=s.get("translated_sentence"),
            )
            for s in data["simplified_text"]["simplified_sentences"]
        ]

        revised_sentences = [
            RevisedSentence(
                sentence=s["sentence"],
                image_prompt=s["image_prompt"],
                highlighted=s["highlighted"],
                translated_sentence=s.get("translated_sentence"),
            )
            for s in data["revision"]["revised_sentences"]
        ]

        return SimplifyResponse(
            title=data["simplified_text"]["title"],
            simplified_sentences=simplified_sentences,
            validation=data["validation"],
            revised_sentences=revised_sentences,
        )

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to the target language."""
        response = requests.post(
            f"{self.base_url}/sentence/translate",
            params={"text": text, "target_language": target_language},
            timeout=60,
        )
        response.raise_for_status()
        return response.text

    def generate_icons(
        self, sentences: list[RevisedSentence]
    ) -> IconsResponse:
        """
        Generate icons for the given sentences.

        Args:
            sentences: List of revised sentences to generate icons for

        Returns:
            IconsResponse with request_id and list of generated icons

        Raises:
            requests.RequestException: If the API request fails
        """
        payload = {
            "sentences": [
                {
                    "sentence": s.sentence,
                    "image_prompt": s.image_prompt,
                    "highlighted": s.highlighted,
                }
                for s in sentences
            ]
        }

        response = requests.post(
            f"{self.base_url}/sentence/generate-icons",
            json=payload,
            timeout=300,  # Icon generation can take longer
        )
        response.raise_for_status()
        data = response.json()

        icons = [
            GeneratedIcon(
                sentence=icon["sentence"],
                image_prompt=icon["image_prompt"],
                highlighted=icon["highlighted"],
                image_path=icon["image_path"],
            )
            for icon in data["icons"]
        ]

        return IconsResponse(
            request_id=data["request_id"],
            icons=icons,
        )

    def search_symbols(self, sentences: list[RevisedSentence]) -> SymbolSearchResponse:
        """Search Global Symbols API for each sentence."""
        payload = {
            "sentences": [
                {
                    "sentence": s.sentence,
                    "image_prompt": s.image_prompt,
                    "highlighted": s.highlighted,
                }
                for s in sentences
            ]
        }

        response = requests.post(
            f"{self.base_url}/sentence/search-symbols",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        results = [
            SymbolSearchResult(
                sentence=r["sentence"],
                image_prompt=r["image_prompt"],
                highlighted=r["highlighted"],
                symbol_found=r["symbol_found"],
                symbol_image_path=r.get("symbol_image_path"),
            )
            for r in data["results"]
        ]

        return SymbolSearchResponse(request_id=data["request_id"], results=results)

    def generate_ai_icons(self, request_id: str, sentences: list) -> IconsResponse:
        """Generate AI icons for sentences, passing through symbol search results."""
        payload = {
            "request_id": request_id,
            "sentences": sentences,
        }

        response = requests.post(
            f"{self.base_url}/sentence/generate-ai-icons",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        icons = [
            GeneratedIcon(
                sentence=icon["sentence"],
                image_prompt=icon["image_prompt"],
                highlighted=icon["highlighted"],
                image_path=icon["image_path"],
            )
            for icon in data["icons"]
        ]

        return IconsResponse(request_id=data["request_id"], icons=icons)

    def get_icon_url(self, request_id: str, image_id: str) -> str:
        """
        Get the full URL for retrieving an icon image.

        Args:
            request_id: The request ID from icon generation
            image_id: The image filename/ID

        Returns:
            Full URL to fetch the icon image
        """
        return f"{self.base_url}/icons/{request_id}/{image_id}"

    def synthesize_audio(
        self,
        sentences: list[str],
        request_id: Optional[str] = None,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> TTSResponse:
        """
        Synthesize a list of sentence strings to WAV audio via the TTS service.

        Returns:
            TTSResponse with request_id and list of audio files (id + filename).
        """
        payload = {
            "sentences": [{"id": i + 1, "text": text} for i, text in enumerate(sentences)],
            "voice": voice,
            "speed": speed,
        }
        if request_id:
            payload["request_id"] = request_id

        response = requests.post(
            f"{self.base_url}/sentence/synthesize",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return TTSResponse(
            request_id=data["request_id"],
            audio_files=[
                TTSAudioFile(id=af["id"], filename=af["filename"])
                for af in data["audio_files"]
            ],
        )

    def fetch_audio(self, request_id: str, filename: str) -> bytes:
        """Fetch a single WAV audio file from the TTS service."""
        response = requests.get(
            f"{self.base_url}/audio/{request_id}/file/{filename}",
            timeout=30,
        )
        response.raise_for_status()
        return response.content

    def get_audio_export_url(self, request_id: str) -> str:
        """Return the URL for downloading the ZIP of all audio files."""
        return f"{self.base_url}/audio/{request_id}/export"

    def fetch_icon(self, request_id: str, image_id: str) -> bytes:
        """
        Fetch an icon image from the backend.

        Args:
            request_id: The request ID from icon generation
            image_id: The image filename/ID

        Returns:
            Image data as bytes

        Raises:
            requests.RequestException: If the API request fails
        """
        url = self.get_icon_url(request_id, image_id)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
