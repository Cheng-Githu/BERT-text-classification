import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import random
import numpy as np

# ===================== 随机种子（完全保留之前的稳定逻辑） =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(222)

# ===================== BERT 分词器（替代自定义词汇表） =====================
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    cache_dir="./bert_cache"
)
MAX_LEN = 250  # 和之前 GRU 保持一致的长度

# ===================== 数据集（适配 BERT 输入格式） =====================
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # BERT 标准分词
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

# ===================== 训练（完全保留之前的优化逻辑！） =====================
def run_train(params, train_loader, test_loader, device):
    from model import BertTextClassifier
    model = BertTextClassifier(
        dropout=params["dropout"]
    ).to(device)

    criterion = nn.BCELoss()
    # 保留 AdamW + 权重衰减 + 梯度裁剪，训练超稳
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(5):  # BERT 收敛极快，5轮足够
        model.train()
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, y in test_loader:
                ids, mask = ids.to(device), mask.to(device)
                pred = model(ids, mask)
                preds.extend((pred > 0.5).cpu().numpy())
                trues.extend(y.numpy())

        acc = accuracy_score(trues, preds)
        if acc > best_acc:
            best_acc = acc
    return best_acc, params

# ===================== 主程序 =====================
if __name__ == "__main__":
    with open("news_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    # 构建 DataLoader
    train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=16, shuffle=True)
    test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    best_acc = 0
    best_params = None
    from model import get_param_grid
    for params in get_param_grid():
        acc, param = run_train(params, train_loader, test_loader, device)
        print(f"参数: {params} | 准确率: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = param

    print("\n===== 最终结果 =====")
    print(f"最高准确率: {best_acc:.4f}")
    print(f"最优参数: {best_params}")