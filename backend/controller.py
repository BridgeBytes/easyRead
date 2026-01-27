from services.gemini import GeminiDriver
from services.storage import StorageDriver
from services.config import Config
from services.icons import IconGenerator
import json
from logging import getLogger

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

    def validate_text(self, original_sentence: str, simplified_sentences: list[dict]) -> str:
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

    def revise_text(self, original_text: str, easy_read_sentences: list, feedback: str) -> str:
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

    def generate_icons(self, sentences: list[dict]) -> list[bytes]:
        pass

