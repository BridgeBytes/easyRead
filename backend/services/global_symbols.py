"""
Service for querying and downloading symbols from the Global Symbols API.

Global Symbols is a public API that provides access to various AAC (Augmentative
and Alternative Communication) symbol libraries like ARASAAC, Mulberry, etc.

API Documentation: https://globalsymbols.com/api
"""

import requests
from pathlib import Path
from typing import Optional, Union
from logging import getLogger

logger = getLogger(__name__)


class GlobalSymbolsService:
    """Query and download symbols from Global Symbols API."""

    BASE_URL = "https://globalsymbols.com/api/v1/labels/search"

    def __init__(self, timeout: int = 10):
        """
        Initialize the Global Symbols service.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()

    def search(
        self,
        query: str,
        symbolset: str = "arasaac",
        language: str = "eng",
        limit: int = 1
    ) -> list[dict]:
        """
        Search for symbols matching a query.

        Args:
            query: Search term (e.g., "happy", "dog", "reading").
            symbolset: Symbol library to search (e.g., "arasaac", "mulberry").
            language: ISO 639-3 language code (default: "eng" for English).
            limit: Maximum number of results to return.

        Returns:
            List of symbol data dictionaries containing image URLs and metadata.
        """
        params = {
            "query": query,
            "symbolset": symbolset,
            "language": language,
            "limit": limit
        }

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching Global Symbols API: {e}")
            return []

    def download_image(self, image_url: str, output_path: Union[str, Path]) -> Optional[Path]:
        """
        Download an image from a URL and save it to the specified path.

        Args:
            image_url: URL of the image to download.
            output_path: Local path where the image should be saved.

        Returns:
            Path to the saved image, or None if download failed.
        """
        output_path = Path(output_path)

        try:
            response = self.session.get(image_url, timeout=self.timeout)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)

            logger.info(f"Downloaded image to {output_path}")
            return output_path

        except requests.RequestException as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return None

    def search_and_download(
        self,
        query: str,
        output_path: Union[str, Path],
        symbolset: str = "arasaac",
        language: str = "eng"
    ) -> Optional[Path]:
        """
        Search for a symbol and download the first matching result.

        Args:
            query: Search term for the symbol.
            output_path: Local path where the image should be saved.
            symbolset: Symbol library to search (default: "arasaac").
            language: ISO 639-3 language code (default: "eng").

        Returns:
            Path to the downloaded image, or None if no symbol found or download failed.
        """
        results = self.search(query, symbolset=symbolset, language=language, limit=1)

        if not results:
            logger.warning(f"No symbols found for query: '{query}'")
            return None

        symbol = results[0]
        image_url = symbol.get("picto", {}).get("image_url")

        if not image_url:
            logger.warning(f"No image URL found for symbol: {symbol}")
            return None

        return self.download_image(image_url, output_path)
