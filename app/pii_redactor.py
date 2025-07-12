import re
import spacy

nlp = spacy.load("en_core_web_lg")

# Regex patterns
PATTERNS = [
    (re.compile(r"\b\d{10,16}\b"), "[REDACTED_ACCOUNT]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b"), "[REDACTED_SWIFT]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"), "[REDACTED_IBAN]"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "[REDACTED_EMAIL]")
]

def redact_text(text: str) -> str:
    redaction_count = 0
    # Rule-based regex
    for pattern, replacement in PATTERNS:
        matches = len(pattern.findall(text))
        redaction_count += matches
        text = pattern.sub(replacement, text)

    # NER-based (names, locations, etc.)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "MONEY"]:
            redaction_count += 1
            text = text.replace(ent.text, f"[REDACTED_{ent.label_}]")

    return text, redaction_count
