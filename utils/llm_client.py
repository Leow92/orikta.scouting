# utils/llm_client.py

from dotenv import load_dotenv
import os
from groq import Groq
from utils.lang import _lang_block
from requests.exceptions import ReadTimeout

# Load .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
_client = Groq(api_key=GROQ_API_KEY)
model_chosen = "openai/gpt-oss-20b"

def _groq_chat(user_content: str, language: str, model= model_chosen) -> str:
    """
    Generate a chat completion using Groq's official Python SDK.
    This function mirrors the previous Ollama streaming logic but
    uses the Groq library instead of manual HTTP calls.
    """
    try:
        # Create the streaming completion
        stream = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _lang_block(language)},
                {"role": "user", "content": user_content},
            ],
            temperature=1,
            stream=True,
        )

        chunks = []
        for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                chunks.append(event.choices[0].delta.content)

        return "".join(chunks).strip()

    except ReadTimeout:
        # Retry once if Groq takes too long
        return _groq_chat(user_content, language, model)
    except Exception as e:
        return f"⚠️ Groq request failed: {e}"
