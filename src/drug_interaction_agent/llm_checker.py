"""
LLM-based secondary check for drug pairs NOT in our curated database.
Results here are NOT clinically verified -- clearly labeled as such.
"""

import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())


def check_pair_with_llm(drug_a, drug_b):
    prompt = f"""You are a clinical pharmacology reference assistant.
A user wants to know if there is a known drug-drug interaction between "{drug_a}" and "{drug_b}".

Respond with ONLY a JSON object, no other text, in this exact format:
{{"interaction": true or false, "severity": "Contraindicated" or "Major" or "Moderate" or "Minor" or "None", "description": "one sentence explanation"}}

If you are not confident about this specific combination, set interaction to false, severity to "Unknown", and description to "Insufficient reliable information -- consult a pharmacist."
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        data = {"interaction": False, "severity": "Unknown", "description": f"Could not get a reliable AI response ({type(e).__name__})."}
    return data
