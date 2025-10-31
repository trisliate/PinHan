"""
轻量级 seq2seq Transformer（用于拼音 -> 汉字 的任务）。
- 基于 PyTorch 的 nn.Transformer 实现
- 提供简单的词表工具（拼音按 token，汉字按字符）
- 模型体积小、参数可配置，适合快速实验和调试
- 20轮训练示例：python model\train_pinyin2hanzi.py --epochs 20 --batch-size 16 --save-dir outputs\small_model --log-file outputs\train_run.log
"""
from typing import List
import math
import json
import torch
import torch.nn as nn


class Vocab:
    """简单的 token <-> id 映射，包含常用的特殊 token。

    用法：
    - 初始化时可传入 tokens 列表
    - encode/decode 提供基础的序列化/反序列化功能
    """
    def __init__(self, tokens: List[str]=None, unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>'):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.token_to_id = {}
        self.id_to_token = []
        if tokens:
            self.add_tokens(tokens)
        # 确保特殊 token 存在
        self._ensure_special(self.pad_token)
        self._ensure_special(self.unk_token)
        self._ensure_special(self.bos_token)
        self._ensure_special(self.eos_token)

    def _ensure_special(self, tok):
        if tok not in self.token_to_id:
            self.id_to_token.append(tok)
            self.token_to_id[tok] = len(self.id_to_token) - 1

    def add_tokens(self, tokens: List[str]):
        for t in tokens:
            if t not in self.token_to_id:
                self.id_to_token.append(t)
                self.token_to_id[t] = len(self.id_to_token) - 1

    def __len__(self):
        return len(self.id_to_token)

    def token_to_idx(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def idx_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return self.unk_token

    def encode(self, tokens: List[str], add_bos_eos=True) -> List[int]:
        ids = [self.token_to_idx(t) for t in tokens]
        if add_bos_eos:
            return [self.token_to_id[self.bos_token]] + ids + [self.token_to_id[self.eos_token]]
        return ids

    def decode(self, ids: List[int], skip_specials=True) -> List[str]:
        toks = [self.idx_to_token(i) for i in ids]
        if skip_specials:
            toks = [t for t in toks if t not in {self.pad_token, self.bos_token, self.eos_token, self.unk_token}]
        return toks

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'tokens': self.id_to_token, 'unk': self.unk_token, 'pad': self.pad_token, 'bos': self.bos_token, 'eos': self.eos_token}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        v = cls()
        v.token_to_id = {t:i for i,t in enumerate(data['tokens'])}
        v.id_to_token = data['tokens']
        v.unk_token = data.get('unk','<unk>')
        v.pad_token = data.get('pad','<pad>')
        v.bos_token = data.get('bos','<s>')
        v.eos_token = data.get('eos','</s>')
        return v


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码模块（支持序列优先 (seq_len, batch, d) 与批次优先 (batch, seq_len, d) 两种输入布局）。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状 (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持两种输入布局：
        # - (batch, seq_len, d_model)
        # - (seq_len, batch, d_model)
        if x.dim() != 3:
            raise ValueError('PositionalEncoding 期望输入为 3 维张量')
        # 若第一个维度等于位置编码的长度，则推测为 (seq_len, batch, d_model)
        if x.size(0) <= self.pe.size(1) and x.size(0) != x.size(1):
            # seq-first: x (S, N, D) -> pe: (S, 1, D) 广播到 (S, N, D)
            pe = self.pe[0, :x.size(0), :].unsqueeze(1)
            x = x + pe
        else:
            # batch-first: x (N, S, D)
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """基于 nn.Transformer 的简单序列到序列模型。

    说明：PyTorch 的 nn.Transformer 接受的输入维度为 (seq_len, batch_size, d_model)，
    因此数据在进入模型前通常需要进行转置。
    本模型允许指定 source 与 target 的 padding idx（可能不同）。
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_encoder_layers: int = 3, num_decoder_layers: int = 3, dim_feedforward: int = 512,
                 dropout: float = 0.1, pad_idx_src: int = 0, pad_idx_tgt: int = 0):
        super().__init__()
        self.d_model = d_model
        # 使用不同的 padding_idx 构建源/目标的 embedding
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx_src)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx_tgt)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # src: (S, N), tgt: (T, N)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        # output 形状: (T, N, d_model)
        logits = self.generator(output)
        return logits


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """生成解码器用的上三角遮盖（防止看到未来 token）。"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
