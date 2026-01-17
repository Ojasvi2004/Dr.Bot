# medical_ner.py
import logging
from typing import List, Dict, Optional

from transformers import pipeline


class MedicalNER:
    """
    Production-stable Medical NER.

    Strategy:
    - MedCAT is OPTIONAL (only if valid model pack provided)
    - BioBERT is DEFAULT (always available)
    - Never crashes pipeline
    """

    def __init__(
        self,
        medcat_model_pack: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)

        self.cat = None
        self.ner_pipeline = None

        # -----------------------------
        # OPTIONAL MedCAT (disabled by default)
        # -----------------------------
        if medcat_model_pack:
            try:
                from medcat.cat import CAT
                self.cat = CAT.load_model_pack(medcat_model_pack)
                self.logger.info(f"MedCAT model pack loaded: {medcat_model_pack}")
            except Exception as e:
                self.logger.error(
                    "MedCAT model pack failed to load. Disabling MedCAT.",
                    exc_info=True
                )
                self.cat = None

        # -----------------------------
        # BioBERT NER (PRIMARY)
        # -----------------------------
        try:
          self.ner_pipeline = pipeline(
    "ner",
    model="kamalkraj/Bio_ClinicalBERT",
    tokenizer="kamalkraj/Bio_ClinicalBERT",
    aggregation_strategy="simple"
)

            self.logger.info("BioBERT BC5CDR disease model loaded successfully.")
        except Exception:
            self.logger.critical(
                "FAILED to load BioBERT model. NER will be unavailable.",
                exc_info=True
            )
            self.ner_pipeline = None

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract medical entities.

        Output format:
        {
            text: str
            type: condition | drug | symptom
            cui: Optional[str]
            confidence: float
            start: int
            end: int
            source: medcat | biobert
        }
        """

        entities: List[Dict] = []

        # -----------------------------
        # MedCAT extraction (if enabled)
        # -----------------------------
        if self.cat:
            try:
                doc = self.cat.get_entities(text)
                for ent in doc["entities"].values():
                    entities.append({
                        "text": ent["source_value"],
                        "type": ent.get("type_name", "condition"),
                        "cui": ent.get("cui"),
                        "confidence": float(ent.get("confidence", 1.0)),
                        "start": ent["start"],
                        "end": ent["end"],
                        "source": "medcat"
                    })

                if entities:
                    return entities

            except Exception:
                self.logger.error("MedCAT extraction failed. Falling back.", exc_info=True)

        # -----------------------------
        # BioBERT extraction (fallback)
        # -----------------------------
        if not self.ner_pipeline:
            return entities

        try:
            results = self.ner_pipeline(text)
            for ent in results:
                entities.append({
                    "text": ent["word"],
                    "type": "condition",
                    "cui": None,
                    "confidence": float(ent["score"]),
                    "start": ent["start"],
                    "end": ent["end"],
                    "source": "biobert"
                })
        except Exception:
            self.logger.error("BioBERT extraction failed.", exc_info=True)

        return entities
if __name__ == "__main__":
    mytext = "My name is Ojasvi. I am 23 years old.I have chest pain,fever and pneumonia."
    scb = MedicalNER()
    print(f"Original: {mytext}")
    print(f"Scrubbed: {scb.extract_entities(mytext)}")
