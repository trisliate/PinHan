"""Seq2Seq Transformer 模型实现，用于拼音到汉字的转换。"""
from typing import List, Optional
import math
import orjson
import torch
import torch.nn as nn


class Vocab:
    """词表：token <-> ID 的双向映射。"""
    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        unk_token: str = '<unk>',
        pad_token: str = '<pad>',
        bos_token: str = '<s>',
        eos_token: str = '</s>'
    ) -> None:
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        if tokens:
            self.add_tokens(tokens)
        for tok in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            self._ensure_special(tok)

    def _ensure_special(self, tok: str) -> None:
        """确保特殊 token 在词表中。"""
        if tok not in self.token_to_id:
            self.id_to_token.append(tok)
            self.token_to_id[tok] = len(self.id_to_token) - 1

    def add_tokens(self, tokens: List[str]) -> None:
        """添加 token."""
        for t in tokens:
            if t not in self.token_to_id:
                self.id_to_token.append(t)
                self.token_to_id[t] = len(self.id_to_token) - 1

    def __len__(self) -> int:
        return len(self.id_to_token)

    def token_to_idx(self, token: str) -> int:
        """Token -> ID."""
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def idx_to_token(self, idx: int) -> str:
        """ID -> Token."""
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return self.unk_token

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        """编码."""
        ids = [self.token_to_idx(t) for t in tokens]
        if add_bos_eos:
            return [self.token_to_id[self.bos_token]] + ids + [self.token_to_id[self.eos_token]]
        return ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> List[str]:
        """解码."""
        toks = [self.idx_to_token(i) for i in ids]
        if skip_specials:
            special_tokens = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
            toks = [t for t in toks if t not in special_tokens]
        return toks

    def save(self, path: str) -> None:
        """保存词表."""
        data = {
            'tokens': self.id_to_token,
            'unk': self.unk_token,
            'pad': self.pad_token,
            'bos': self.bos_token,
            'eos': self.eos_token,
        }
        with open(path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, path: str) -> 'Vocab':
        """加载词表."""
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
        v = cls()
        v.token_to_id = {t: i for i, t in enumerate(data['tokens'])}
        v.id_to_token = data['tokens']
        v.unk_token = data.get('unk', '<unk>')
        v.pad_token = data.get('pad', '<pad>')
        v.bos_token = data.get('bos', '<s>')
        v.eos_token = data.get('eos', '</s>')
        return v


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用位置编码."""
        if x.size(0) <= self.pe.size(1) and x.size(0) != x.size(1):
            pe = self.pe[0, :x.size(0), :].unsqueeze(1)
            x = x + pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Seq2Seq Transformer 模型."""
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx_src: int = 0,
        pad_idx_tgt: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx_src)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx_tgt)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # 保持False, 因为我们手动处理序列维度
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播."""
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(output)
        return logits


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """生成因果掩码."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask