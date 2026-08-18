"""
Differential Diagnosis Agent -- LLM-based reasoning (Groq).
"""

import os
import json
from groq import Groq
from json_repair import repair_json

client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())


def get_differential_diagnosis(symptoms, age=None, sex=None, history=None):
    context_parts = [f"Symptoms: {symptoms}"]
    if age:
        context_parts.append(f"Age: {age}")
    if sex:
        context_parts.append(f"Sex: {sex}")
    if history:
        context_parts.append(f"Relevant history: {history}")
    context = "\n".join(context_parts)

    prompt = f"""You are a clinical reasoning assistant helping generate a
differential diagnosis list (like a doctor's initial working list of
possibilities before further testing) -- NOT a final diagnosis.

Patient information:
{context}

Respond with ONLY a JSON object, no other text, in this exact format:
{{
  "differential": [
    {{"diagnosis": "...", "likelihood": "High" or "Medium" or "Low", "reasoning": "one sentence"}},
    ...
  ]
}}

List 3-5 possibilities, ordered from most to least likely. Be conservative and clinically reasonable -- favor common conditions unless symptoms strongly suggest otherwise."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            repaired = repair_json(text)
            data = json.loads(repaired)
        return data
    except (json.JSONDecodeError, Exception) as e:
        return {"differential": [], "error": f"Could not get a reliable response ({type(e).__name__})"}


if __name__ == "__main__":
    result = get_differential_diagnosis(
        symptoms="fever, cough, chest pain, shortness of breath for 3 days",
        age=45, sex="M"
    )
    for item in result.get("differential", []):
        print(item)
    if "error" in result:
        print("Error:", result["error"])
