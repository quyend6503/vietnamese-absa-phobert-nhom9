# utils/predictor.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.phobert_model import load_model
from utils.constants import (
    MODEL_PATH, MAX_LEN,
    LABEL_COLS, SENTIMENT_MAP, ASPECT_VI_MAP
)
from utils.preprocessing import preprocess_text


class ABSAPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = load_model(MODEL_PATH, device=self.device)

    def predict(self, raw_text: str) -> list:
        """
        Nhận câu đánh giá thô → trả về danh sách các aspect được đề cập
        """
        if not raw_text or not raw_text.strip():
            return []

        # Bước 1: Tiền xử lý (ViTokenizer)
        processed = preprocess_text(raw_text)

        # Bước 2: Tokenize
        encoding = self.tokenizer(
            processed,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Bước 3: Dự đoán
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs["logits"]  # shape: (1, 12, 4)

        # Bước 4: Softmax + Argmax → lấy nhãn có xác suất cao nhất
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (12, 4)
        preds = probs.argmax(axis=-1)                        # (12,)

        # Bước 5: Trả kết quả, bỏ qua nhãn 0 (Không đề cập)
        results = []
        for i, aspect in enumerate(LABEL_COLS):
            pred_id = int(preds[i])
            if pred_id == 0:
                continue
            results.append({
                "aspect_en":  aspect,
                "aspect_vi":  ASPECT_VI_MAP[aspect],
                "sentiment":  SENTIMENT_MAP[pred_id],
                "confidence": round(float(probs[i, pred_id]) * 100, 2)
            })

        return results