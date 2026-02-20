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


        

