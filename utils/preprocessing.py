# utils/preprocessing.py

# pyrefly: ignore [missing-import]
from pyvi import ViTokenizer


def preprocess_text(text: str) -> str:
    """
    Tiền xử lý giống lúc train PhoBERT
    """

    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    return ViTokenizer.tokenize(text)