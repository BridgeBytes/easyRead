from services.gemini import GeminiDriver
from services.storage import StorageDriver
from services.config import Config
from services.icons import IconGenerator
import json
import os
from logging import getLogger
from uuid import uuid4

logger = getLogger(__name__)


class Controller:

    def __init__(self):
        self.gemini = GeminiDriver()
        self.storage = StorageDriver()
        self.config = Config()
        self.icon_generator = IconGenerator()


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

    def generate_icons(self, sentences: list[dict]) -> dict:
        """
        Logic:
            * Generate a unique request id string using uuid4
            * Recieve List of sentences with image prompts in dict format
            * For each sentence, generate icon using image prompt
                * assign the image to a variable
                * using the temp dir locally check if the subfolder using the request id string exists, if not create it using os.makedirs
                * save the image to the subfolder with name being the image prompted join with _ using pillow image.save(path)
        """

        request_id = str(uuid4())
        for sentence in sentences:
            prompt = sentence['image_prompt']
            image = self.icon_generator.generate(prompt)

            # Create directory for the request if it doesn't exist
            request_dir = os.path.join(self.config.ICON_OUTPUT_PATH, request_id)
            os.makedirs(request_dir, exist_ok=True)

            # Save the image with a filename based on the prompt
            safe_prompt = "_".join(prompt.split())  # Simple way to make filename safe
            image_path = os.path.join(request_dir, f"{safe_prompt}.png")
            image.save(image_path)
            logger.info(f"Saved icon for prompt '{prompt}' at '{image_path}'")
            sentence['image_path'] = "/".join([request_id, f"{safe_prompt}.png"])

        return {
            "request_id": request_id,
            "icons": sentences
        }


        

