import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# ===========================================================================
# 1. LỚP TÍNH LOSS NÂNG CAO 
# ===========================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            focal_weight = (1.0 - probs).pow(self.gamma)
            loss = -(focal_weight * log_probs * smooth_targets).sum(dim=-1)
        else:
            log_probs = F.log_softmax(inputs, dim=-1)
            probs = torch.exp(log_probs)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            p_t = (probs * targets_one_hot).sum(dim=-1)
            focal_weight = (1.0 - p_t).pow(self.gamma)
            loss = focal_weight * F.cross_entropy(inputs, targets, reduction='none')

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        return loss.mean()

# ===========================================================================
# 2. KIẾN TRÚC MẠNG MULTI-HEAD HOÀN CHỈNH
# ===========================================================================
class PhoBERT_ABSA_MultiHead(nn.Module):
    def __init__(self, model_name_or_path, num_aspects=12, num_classes=4,
                 dropout_rate=0.2, num_dropout_samples=5):
        super().__init__()
        self.num_aspects = num_aspects
        self.num_classes = num_classes
        self.num_dropout_samples = num_dropout_samples

        self.phobert = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.phobert.config.hidden_size  

        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_dropout_samples)
        ])
        self.aspect_heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        shared_features = self.shared_dense(cls_output)

        all_logits = []
        for head in self.aspect_heads:
            if self.training and self.num_dropout_samples > 1:
                head_logits = torch.stack([
                    head(dp(shared_features)) for dp in self.dropouts
                ], dim=0).mean(dim=0)
            else:
                head_logits = head(self.dropouts[0](shared_features))
            all_logits.append(head_logits)

        logits = torch.stack(all_logits, dim=1)  # (batch, 12, 4)

        loss = None
        if labels is not None:
            loss = 0.0
            for i in range(self.num_aspects):
                alpha = self.aspect_class_weights[i] if hasattr(self, 'aspect_class_weights') else None
                focal = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
                loss += focal(logits[:, i, :], labels[:, i])
            loss = loss / self.num_aspects

        return {'loss': loss, 'logits': logits}

# ===========================================================================
# 3. HÀM LOAD MODEL DÙNG CHÍNH XÁC CONFIG 
# ===========================================================================
def load_model(model_path: str, device=None):
    import json, os
    from safetensors.torch import load_file

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(model_path, "config_absa.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("CONFIG PATH:", config_path)
    print("CONFIG:", config)
    print("MODEL NAME:", config["model_name_or_path"])

    model = PhoBERT_ABSA_MultiHead(
        model_name_or_path=config["model_name_or_path"],
        num_aspects=config["num_aspects"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        num_dropout_samples=config["num_dropout_samples"],
    )

    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test thử xem file đọc config.json của bạn có mượt không
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # Thử dựng mô hình
        model = load_model(config_data, model_path="model.safetensors")
        print("Chạy thử trên VS Code thành công!")
    except Exception as e:
        print(f"Lưu ý khi chạy thử độc lập: {e}")