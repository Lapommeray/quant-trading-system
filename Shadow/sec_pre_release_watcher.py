
from transformers import pipeline

class SEC_NLP_Spy:
    def __init__(self):
        self.nlp = pipeline("text-classification", model="SEC_BERT")

    def compare_drafts(self, old_draft, new_draft):
        changes = self.nlp(f"DIFF: {old_draft} -> {new_draft}")
        if "compliance_shift" in changes:
            return "ADJUST_PORTFOLIO"
        return "IGNORE"
