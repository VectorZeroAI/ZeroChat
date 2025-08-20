# ZeroMain.py
"""
This is the main chatloop of the app.
It can be run for CLI I/O, or the GUI can use it as backend.
Is fully config defined, so if its not working, first thing to check is the config.py file.
"""

import ZeroMemory
import requests
import json
from config import LLM_SOURSE, OPENROUTER_API_KEY, LOCAL_MODEL_PATH, GEMINI_API_KEY

# LLM CALLS

def call_LLM(prompt: str) -> str:
    if LLM_SOURSE == "OPENROUTER":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "messages": [{"role": "user", "content": prompt}],
        }
        r = requests.post(url, headers=headers, json=data)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    elif LLM_SOURSE == "LOCAL":
        # HuggingFace transformers local pipeline
        from transformers import pipeline
        local_llm = pipeline("text-generation", model=LOCAL_MODEL_PATH)
        result = local_llm(prompt, max_length=512, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]

    elif LLM_SOURSE == "GEMINI":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text

    else:
        raise ValueError("Invalid config: LLM_SOURSE not recognized.")


def call_LLM_with_memory(prompt: str) -> str:
    """
    Memory-augmented LLM call.
    memory.full(prompt) gives related context WITHOUT the prompt itself.
    We need to combine both.
    """
    mem_context = memory.full(prompt)
    full_input = f"Context:\n{mem_context}\n\nUser Prompt:\n{prompt}"
    return call_LLM(full_input)


# ------------------------
# MAIN CHATLOOP
# ------------------------
if __name__ == "__main__":
    print("ZeroMain chatloop started. Type 'exit' to quit.")
    while True:
        user_in = input("You: ")
        if user_in.lower() in ["exit", "quit"]:
            break
        reply = call_LLM_with_memory(user_in)
        print("AI:", reply)