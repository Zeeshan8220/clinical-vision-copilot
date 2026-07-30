"""
Core logic for the Drug Interaction Agent.
Uses a small, curated CSV lookup (rule-based, not ML) -- drug
interactions are known clinical facts, not predictions.
"""

import csv
import os
from itertools import combinations

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "drug_interactions.csv")

ALIASES = {
    "coumadin": "warfarin", "advil": "ibuprofen", "motrin": "ibuprofen",
    "tylenol": "acetaminophen", "glucophage": "metformin", "zocor": "simvastatin",
    "viagra": "sildenafil", "prozac": "fluoxetine", "plavix": "clopidogrel",
    "prilosec": "omeprazole", "lasix": "furosemide", "prinivil": "lisinopril",
    "zestril": "lisinopril",
}


def normalize(name):
    name = name.strip().lower()
    return ALIASES.get(name, name)


def load_interactions():
    interactions = {}
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = frozenset([row["drug_a"], row["drug_b"]])
            interactions[key] = {"severity": row["severity"], "description": row["description"]}
    return interactions


INTERACTIONS_DB = load_interactions()


def check_interactions(medication_list):
    normalized = [normalize(m) for m in medication_list]
    flagged = []
    for drug_a, drug_b in combinations(normalized, 2):
        key = frozenset([drug_a, drug_b])
        if key in INTERACTIONS_DB:
            info = INTERACTIONS_DB[key]
            flagged.append({"drug_a": drug_a, "drug_b": drug_b, "severity": info["severity"], "description": info["description"]})

    severity_order = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3}
    flagged.sort(key=lambda x: severity_order.get(x["severity"], 99))
    return flagged


def check_interactions_hybrid(medication_list, use_llm=True):
    normalized = [normalize(m) for m in medication_list]
    results = []

    for drug_a, drug_b in combinations(normalized, 2):
        key = frozenset([drug_a, drug_b])
        if key in INTERACTIONS_DB:
            info = INTERACTIONS_DB[key]
            results.append({
                "drug_a": drug_a, "drug_b": drug_b,
                "severity": info["severity"], "description": info["description"],
                "source": "verified_database",
            })
        elif use_llm:
            from llm_checker import check_pair_with_llm
            llm_result = check_pair_with_llm(drug_a, drug_b)
            if llm_result.get("interaction"):
                results.append({
                    "drug_a": drug_a, "drug_b": drug_b,
                    "severity": llm_result.get("severity", "Unknown"),
                    "description": llm_result.get("description", ""),
                    "source": "ai_generated_unverified",
                })

    severity_order = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}
    results.sort(key=lambda x: severity_order.get(x["severity"], 99))
    return results


if __name__ == "__main__":
    test_meds = ["Warfarin", "Advil", "Metformin"]
    print(f"Checking: {test_meds}")
    results = check_interactions(test_meds)
    for r in results:
        print(r)
    if not results:
        print("No known interactions found among these medications.")
