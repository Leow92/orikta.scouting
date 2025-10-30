# test_groq.py
from utils.llm_client import _groq_chat

print("Testing Groq connection...")
response = _groq_chat("Say hello in one short sentence.", language="English")
print("Response:\n", response)
