"""
SLM 语义语言模型

轻量级语言模型，用于评估候选句子的流畅度和语义合理性
用于 P2H 候选结果的重排序
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SLMConfig:
    """SLM 模型配置"""
    vocab_size: int = 8000           # 汉字词表大小
    d_model: int = 256               # 隐藏层维度
    n_heads: int = 4                 # 注意力头数
    n_layers: int = 4                # Transformer 层数
    d_ff: int = 1024                 # FFN 维度
    max_len: int = 128               # 最大序列长度
    dropout: float = 0.1
    
    # 特殊 token
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


class SLModel(nn.Module):
    """
    语义语言模型 (Semantic Language Model)
    
    基于 Transformer Decoder 的自回归语言模型
    用于计算句子的困惑度 (perplexity)，评估句子流畅度
    """
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id
        )
        
        # 位置编码
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.n_layers)
        
        # 输出层（与嵌入层共享权重）
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight  # 权重绑定
        
        # Layer Norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,       # [batch, seq_len]
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置 ID
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        x = self.dropout(x)
        
        # 因果掩码
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Padding 掩码
        if attention_mask is None:
            key_padding_mask = (input_ids == self.config.pad_id)
        else:
            key_padding_mask = ~attention_mask.bool()
        
        # Transformer（decoder-only，memory 为空）
        # 使用自注意力，tgt 和 memory 都是 x
        memory = torch.zeros(batch_size, 1, self.config.d_model, device=device)
        x = self.transformer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=key_padding_mask)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 输出投影
        logits = self.output_projection(x)
        
        return logits
    
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果注意力掩码"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        计算语言模型损失（用于评估）
        
        Args:
            input_ids: [batch, seq_len] 输入序列（包含 BOS）
            reduction: 'mean', 'sum', 'none'
        
        Returns:
            loss: 交叉熵损失
        """
        self.eval()
        
        # 输入是 x[:-1]，目标是 x[1:]
        logits = self(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.config.pad_id,
            reduction=reduction
        )
        
        return loss
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算困惑度 (Perplexity)
        
        困惑度越低，句子越流畅
        
        Returns:
            perplexity: [batch] 每个句子的困惑度
        """
        self.eval()
        
        batch_size, seq_len = input_ids.shape
        
        # 计算每个 token 的损失
        logits = self(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        
        # 逐 token 交叉熵
        log_probs = F.log_softmax(logits, dim=-1)
        token_losses = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        
        # 掩码 padding
        mask = (targets != self.config.pad_id).float()
        token_losses = token_losses * mask
        
        # 计算平均损失
        seq_lens = mask.sum(dim=1)
        avg_loss = token_losses.sum(dim=1) / seq_lens.clamp(min=1)
        
        # 困惑度 = exp(avg_loss)
        perplexity = torch.exp(avg_loss)
        
        return perplexity
    
    @torch.no_grad()
    def score_sentences(
        self,
        sentences: List[str],
        vocab: 'SLMVocab',
        device: torch.device = None,
    ) -> List[float]:
        """
        评分句子列表
        
        Args:
            sentences: 句子列表
            vocab: 词表
            device: 设备
        
        Returns:
            scores: 分数列表（负困惑度，越高越好）
        """
        self.eval()
        device = device or next(self.parameters()).device
        
        # 编码
        encoded = []
        max_len = 0
        for sent in sentences:
            ids = [self.config.bos_id] + vocab.encode(sent) + [self.config.eos_id]
            encoded.append(ids)
            max_len = max(max_len, len(ids))
        
        # Padding
        padded = []
        for ids in encoded:
            padded.append(ids + [self.config.pad_id] * (max_len - len(ids)))
        
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        
        # 计算困惑度
        ppl = self.compute_perplexity(input_ids)
        
        # 返回负困惑度作为分数（越高越好）
        scores = (-ppl).tolist()
        
        return scores


class SLMVocab:
    """SLM 词表"""
    
    def __init__(self):
        self.char2id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.id2char = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
        
        # 特殊 token ID
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
    
    def build_from_freq(self, char_freq: dict, max_vocab: int = 8000, min_freq: float = 0.0):
        """从频率字典构建词表"""
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        for char, freq in sorted_chars[:max_vocab - 4]:
            if freq >= min_freq and char not in self.char2id:
                idx = len(self.char2id)
                self.char2id[char] = idx
                self.id2char[idx] = char
    
    def encode(self, text: str) -> List[int]:
        """文本 -> ID 序列"""
        return [self.char2id.get(c, self.char2id['<unk>']) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        """ID 序列 -> 文本"""
        chars = []
        for id in ids:
            if id in (0, 1):  # pad, bos
                continue
            if id == 2:  # eos
                break
            chars.append(self.id2char.get(id, '?'))
        return ''.join(chars)
    
    @property
    def vocab_size(self) -> int:
        return len(self.char2id)
    
    def save(self, path: str):
        """保存词表"""
        import orjson
        with open(path, 'wb') as f:
            f.write(orjson.dumps({'char2id': self.char2id}))
    
    def load(self, path: str):
        """加载词表"""
        import orjson
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
        self.char2id = data['char2id']
        self.id2char = {int(v): k for k, v in self.char2id.items()}
        
        # 重新设置特殊 token ID
        self.pad_id = self.char2id.get('<pad>', 0)
        self.bos_id = self.char2id.get('<bos>', 1)
        self.eos_id = self.char2id.get('<eos>', 2)
        self.unk_id = self.char2id.get('<unk>', 3)


class CandidateReranker:
    """候选重排序器"""
    
    def __init__(self, slm: SLModel, vocab: SLMVocab, device: torch.device = None):
        self.slm = slm
        self.vocab = vocab
        self.device = device or next(slm.parameters()).device
        self.slm.to(self.device)
        self.slm.eval()
    
    def rerank(
        self,
        candidates: List[str],
        p2h_scores: List[float] = None,
        alpha: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        重排序候选
        
        Args:
            candidates: 候选句子列表
            p2h_scores: P2H 模型给出的分数（可选）
            alpha: P2H 分数权重（1-alpha 为 SLM 权重）
        
        Returns:
            重排序后的 (句子, 综合分数) 列表
        """
        if not candidates:
            return []
        
        # 计算 SLM 分数
        slm_scores = self.slm.score_sentences(candidates, self.vocab, self.device)
        
        # 归一化 SLM 分数到 0-1
        min_s, max_s = min(slm_scores), max(slm_scores)
        if max_s > min_s:
            slm_scores_norm = [(s - min_s) / (max_s - min_s) for s in slm_scores]
        else:
            slm_scores_norm = [0.5] * len(slm_scores)
        
        # 综合分数
        if p2h_scores:
            # 归一化 P2H 分数
            min_p, max_p = min(p2h_scores), max(p2h_scores)
            if max_p > min_p:
                p2h_norm = [(s - min_p) / (max_p - min_p) for s in p2h_scores]
            else:
                p2h_norm = [0.5] * len(p2h_scores)
            
            final_scores = [
                alpha * p + (1 - alpha) * s
                for p, s in zip(p2h_norm, slm_scores_norm)
            ]
        else:
            final_scores = slm_scores_norm
        
        # 排序
        results = list(zip(candidates, final_scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


if __name__ == '__main__':
    # 测试模型结构
    config = SLMConfig()
    model = SLModel(config)
    
    print(f"SLM 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 20
    
    input_ids = torch.randint(4, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    print(f"输入: {input_ids.shape}")
    print(f"输出 logits: {logits.shape}")
    
    # 测试困惑度计算
    ppl = model.compute_perplexity(input_ids)
    print(f"困惑度: {ppl}")
