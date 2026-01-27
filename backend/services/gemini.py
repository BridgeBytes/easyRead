from google import genai

class GeminiDriver:
    def __init__(self):
        self.client = genai.Client()

    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview", contents=prompt
        )
        return response.text