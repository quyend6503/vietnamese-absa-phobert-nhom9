# utils/predictor.py

import torch
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
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

    def _run_inference(self, raw_text: str):
        """
        Chạy inference và trả về (probs, preds) cho tất cả 12 aspect.
        probs shape: (12, 4)  |  preds shape: (12,)
        """
        if not raw_text or not raw_text.strip():
            return None, None

        processed = preprocess_text(raw_text)

        encoding = self.tokenizer(
            processed,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs["logits"]  # (1, 12, 4)

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (12, 4)
        preds = probs.argmax(axis=-1)                        # (12,)
        return probs, preds

    def predict(self, raw_text: str) -> list:
        """
        Trả về danh sách các aspect ĐƯỢC ĐỀ CẬP (bỏ nhãn 0).
        Tương thích ngược với code cũ.
        """
        probs, preds = self._run_inference(raw_text)
        if probs is None:
            return []

        results = []
        for i, aspect in enumerate(LABEL_COLS):
            pred_id = int(preds[i])
            if pred_id == 0:
                continue
            results.append({
                "aspect_en":  aspect,
                "aspect_vi":  ASPECT_VI_MAP[aspect],
                "sentiment":  SENTIMENT_MAP[pred_id],
                "sentiment_id": pred_id,
                "confidence": round(float(probs[i, pred_id]) * 100, 2)
            })
        return results

    def predict_full(self, raw_text: str) -> list:
        """
        Trả về TẤT CẢ 12 aspect (kể cả nhãn 0 — Không đề cập).
        Dùng cho Streamlit để hiển thị đầy đủ 12 card.
        """
        probs, preds = self._run_inference(raw_text)
        if probs is None:
            return []

        results = []
        for i, aspect in enumerate(LABEL_COLS):
            pred_id = int(preds[i])
            results.append({
                "aspect_en":    aspect,
                "aspect_vi":    ASPECT_VI_MAP[aspect],
                "sentiment":    SENTIMENT_MAP[pred_id],
                "sentiment_id": pred_id,                            # 0/1/2/3
                "confidence":   round(float(probs[i, pred_id]) * 100, 2),
                "prob_none":    round(float(probs[i, 0]) * 100, 2),
                "prob_pos":     round(float(probs[i, 1]) * 100, 2),
                "prob_neg":     round(float(probs[i, 2]) * 100, 2),
                "prob_neu":     round(float(probs[i, 3]) * 100, 2),
                "mentioned":    pred_id != 0
            })
        return results