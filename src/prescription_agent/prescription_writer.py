"""
Prescription Writer Agent -- generates a DRAFT prescription for
physician review (not a final prescription).
"""

import os
import json
from groq import Groq
from json_repair import repair_json

client = Groq(api_key=os.environ.get("GROQ_API_KEY", "").strip())


def generate_prescription_draft(diagnosis, age=None, sex=None, allergies=None, current_medications=None):
    context_parts = [f"Diagnosis: {diagnosis}"]
    if age:
        context_parts.append(f"Age: {age}")
    if sex:
        context_parts.append(f"Sex: {sex}")
    if allergies:
        context_parts.append(f"Known allergies: {allergies}")
    if current_medications:
        context_parts.append(f"Current medications: {current_medications}")
    context = "\n".join(context_parts)

    prompt = f"""You are a clinical documentation assistant helping a doctor
draft a prescription. This is a DRAFT ONLY -- a licensed physician will
review, edit, and sign it before it is used.

Patient information:
{context}

Respond with ONLY a JSON object, no other text, in this exact format:
{{
  "medications": [
    {{"name": "...", "dosage": "...", "frequency": "...", "duration": "...", "notes": "..."}}
  ],
  "follow_up": "when the patient should be seen again and why",
  "referral": "specialist referral if warranted, or 'None needed' otherwise",
  "red_flags": "symptoms that should prompt immediate return/emergency care"
}}

Be conservative and general (typical first-line treatment for common presentations of this diagnosis). If the diagnosis is vague, say so in the notes field rather than guessing specifics."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = json.loads(repair_json(text))
        return data
    except Exception as e:
        return {"error": f"Could not generate draft ({type(e).__name__})"}


if __name__ == "__main__":
    result = generate_prescription_draft(
        diagnosis="Community-acquired pneumonia (mild, outpatient)",
        age=45, sex="M", allergies="None known", current_medications="None"
    )
    print(json.dumps(result, indent=2))
