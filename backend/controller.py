from services.gemini import GeminiDriver
from services.storage import StorageDriver
from services.config import Config
from services.icons import IconGenerator
from services.global_symbols import GlobalSymbolsService
import json
import os
from logging import getLogger
from uuid import uuid4
from pathlib import Path

logger = getLogger(__name__)


class Controller:

    def __init__(self):
        self.gemini = GeminiDriver()
        self.storage = StorageDriver()
        self.config = Config()
        self.icon_generator = IconGenerator()
        self.global_symbols = GlobalSymbolsService()


    def simplify_text(self, text: str) -> dict:
        template = self.config.simplify_text['system_message']
        prompt = f"{template}\n\n Bellow is the Input Text to simplify:\n\n{text}\n\n"

        response = self.gemini.generate_text(prompt)

        try:
            response_data = json.loads(response)
            logger.info(f"Successfully parsed response JSON: {response_data}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON. Raw response: {response}")
            response_data = {"error": "Failed to parse response as JSON.", "raw_response": response}
        return response_data

    def validate_text(self, original_sentence: str, simplified_sentences: list[dict]) -> dict:
        template = self.config.validate_text['system_message']
        prompt = template + "\n" + self.config.validate_text["user_message_template"].format(original_markdown=original_sentence, simplified_sentences=json.dumps(simplified_sentences))
        response = self.gemini.generate_text(prompt)

        try:
            response_data = json.loads(response)
            logger.info(f"Successfully parsed response JSON: {response_data}")
        except json.JSONDecodeError:
            response_data = {"error": "Failed to parse response as JSON.", "raw_response": response}
            logger.error(f"Failed to parse response as JSON. Raw response: {response}")
        return response_data

    def revise_text(self, original_text: str, easy_read_sentences: list, feedback: str) -> dict:
        template = self.config.revise_text['system_message']
        prompt = template + "\n" + self.config.revise_text["user_message_template"].format(original_markdown=original_text, simplified_sentences=json.dumps(easy_read_sentences), validation_feedback=feedback)
        response = self.gemini.generate_text(prompt)

        try:
            response_data = json.loads(response)
            logger.info(f"Successfully parsed response JSON: {response_data}")
        except json.JSONDecodeError:
            response_data = {"error": "Failed to parse response as JSON.", "raw_response": response}
            logger.error(f"Failed to parse response as JSON. Raw response: {response}")

        return response_data

    def generate_icons(self, sentences: list[dict], symbolset: str = "arasaac", use_global_symbols: bool = True) -> dict:
        """
        Generate icons for sentences using Global Symbols API or fallback to local generation.

        Logic:
            * Generate a unique request id string using uuid4
            * Receive List of sentences with image prompts in dict format
            * For each sentence:
                * First, try to query Global Symbols API with the image prompt
                * If found, download and save the symbol image
                * If not found and fallback enabled, generate icon using local LoRA model
                * Save the image to request-specific subfolder

        Args:
            sentences: List of sentence dicts with 'image_prompt' field
            symbolset: Symbol library to query (default: "arasaac")
            use_global_symbols: Whether to use Global Symbols API (default: True)

        Returns:
            Dict with request_id and list of icons with image_path added
        """

        request_id = str(uuid4())
        request_dir = Path(self.config.ICON_OUTPUT_PATH) / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        for sentence in sentences:
            prompt = sentence['image_prompt']
            safe_prompt = "_".join(prompt.split())  # Simple way to make filename safe
            image_path = request_dir / f"{safe_prompt}.png"

            image_found = False

            # Try Global Symbols API first if enabled
            if use_global_symbols:
                logger.info(f"Querying Global Symbols API for prompt: '{prompt}'")
                try:
                    downloaded_path = self.global_symbols.search_and_download(
                        query=prompt,
                        output_path=image_path,
                        symbolset=symbolset
                    )

                    if downloaded_path:
                        logger.info(f"Successfully retrieved symbol from Global Symbols API for '{prompt}'")
                        image_found = True
                    else:
                        logger.warning(f"No symbol found in Global Symbols API for '{prompt}'")

                except Exception as e:
                    logger.error(f"Error querying Global Symbols API for '{prompt}': {e}")

            # Fallback to local icon generation if Global Symbols didn't work
            if not image_found:
                logger.info(f"Generating icon locally for prompt: '{prompt}'")
                try:
                    image = self.icon_generator.generate(prompt)
                    image.save(image_path)
                    logger.info(f"Saved locally generated icon for '{prompt}' at '{image_path}'")
                except Exception as e:
                    logger.error(f"Error generating icon locally for '{prompt}': {e}")
                    continue

            # Set the relative image path for the response
            sentence['image_path'] = "/".join(["icons", request_id, f"{safe_prompt}.png"])

        return {
            "request_id": request_id,
            "icons": sentences
        }

    def search_symbols(self, sentences: list[dict], symbolset: str = "arasaac") -> dict:
        """
        Search Global Symbols API for each sentence without falling back to AI generation.

        Returns:
            Dict with request_id and list of results containing symbol_found + symbol_image_path.
        """
        request_id = str(uuid4())
        request_dir = Path(self.config.ICON_OUTPUT_PATH) / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for sentence in sentences:
            prompt = sentence['image_prompt']
            safe_prompt = "_".join(prompt.split())
            image_path = request_dir / f"{safe_prompt}.png"

            symbol_found = False
            symbol_image_path = None

            try:
                downloaded_path = self.global_symbols.search_and_download(
                    query=prompt,
                    output_path=image_path,
                    symbolset=symbolset
                )
                if downloaded_path:
                    symbol_found = True
                    symbol_image_path = "/".join(["icons", request_id, f"{safe_prompt}.png"])
                    logger.info(f"Found symbol for '{prompt}'")
                else:
                    logger.info(f"No symbol found for '{prompt}'")
            except Exception as e:
                logger.error(f"Error searching symbols for '{prompt}': {e}")

            results.append({
                "sentence": sentence['sentence'],
                "image_prompt": sentence['image_prompt'],
                "highlighted": sentence['highlighted'],
                "symbol_found": symbol_found,
                "symbol_image_path": symbol_image_path,
            })

        return {
            "request_id": request_id,
            "results": results,
        }

    def generate_ai_icons(self, request_id: str, sentences: list[dict]) -> dict:
        """
        Generate AI icons for sentences, passing through symbol search results where available.

        For sentences with symbol_image_path set, uses that path directly.
        For others, generates via the ETH LoRA model using ai_prompt.

        Returns:
            Dict with request_id and list of icons matching GenerateIconsResponse shape.
        """
        request_dir = Path(self.config.ICON_OUTPUT_PATH) / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        for sentence in sentences:
            symbol_image_path = sentence.get('symbol_image_path')

            if symbol_image_path:
                sentence['image_path'] = symbol_image_path
                logger.info(f"Using existing symbol for '{sentence['sentence']}'")
            else:
                ai_prompt = sentence['ai_prompt']
                safe_prompt = "_".join(ai_prompt.split())[:80]
                image_path = request_dir / f"ai_{safe_prompt}.png"

                try:
                    image = self.icon_generator.generate(ai_prompt)
                    image.save(image_path)
                    logger.info(f"Saved AI-generated icon for '{ai_prompt}'")
                except Exception as e:
                    logger.error(f"Error generating AI icon for '{ai_prompt}': {e}")
                    continue

                sentence['image_path'] = "/".join(["icons", request_id, f"ai_{safe_prompt}.png"])

            # Keep image_prompt for response compatibility with GenerateIconsResponse
            sentence['image_prompt'] = sentence.get('ai_prompt', sentence.get('sentence', ''))

        return {
            "request_id": request_id,
            "icons": sentences,
        }
