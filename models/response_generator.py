import os
import requests
import json

class ResponseGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, review: str, sentiment: str) -> str:
        prompt = f"""Recensione: "{review}"
Sentiment: {sentiment}
Genera una risposta cortese e professionale, empatica se negativa, che possa essere usata come risposta pubblica da parte dell'attività su Google Recensioni. Scrivi in italiano."""

        data = {
            "model": "anthropic/claude-sonnet-4",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"""Recensione: "{review}" """}
            ]
        }

        response = requests.post(self.base_url, headers=self.headers, json=data)

        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Errore generazione risposta]: {e}"
