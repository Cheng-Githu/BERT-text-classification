import torch
import torch.nn as nn
from transformers import BertModel


class BertTextClassifier(nn.Module):  # 这里修复了！之前少打了一个n
    def __init__(self, dropout=0.3):
        super().__init__()

        # 加载预训练 BERT
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            cache_dir="./bert_cache"
        )

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 1)  # BERT-base 输出维度固定 768
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # BERT 前向传播
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = bert_out.pooler_output  # 取 <CLS> 句子特征

        # 分类输出
        feat = self.dropout(cls_feat)
        out = self.fc(feat)
        return self.sigmoid(out).squeeze()


def get_param_grid():
    # BERT 微调专属超参数（小学习率！）
    return [
        {"dropout": 0.3, "lr": 2e-5},
        {"dropout": 0.5, "lr": 1e-5},
    ]