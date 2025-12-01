"""
P2H 主模型 (Pinyin-to-Hanzi)

基于 Transformer 的序列到序列模型
将拼音序列转换为汉字序列
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class P2HConfig:
    """P2H 模型配置"""
    # 词表大小
    pinyin_vocab_size: int = 500     # 拼音词表（约430个有效拼音 + 特殊token）
    hanzi_vocab_size: int = 8000     # 汉字词表（常用汉字）
    
    # 模型维度
    d_model: int = 256               # 隐藏层维度
    n_heads: int = 4                 # 注意力头数
    n_encoder_layers: int = 4        # 编码器层数
    n_decoder_layers: int = 4        # 解码器层数
    d_ff: int = 1024                 # FFN 维度
    
    # 序列长度
    max_pinyin_len: int = 64         # 最大拼音序列长度
    max_hanzi_len: int = 64          # 最大汉字序列长度
    
    # 正则化
    dropout: float = 0.1
    
    # 特殊 token
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class P2HModel(nn.Module):
    """
    Pinyin-to-Hanzi 序列到序列模型
    
    编码器：处理拼音序列
    解码器：生成汉字序列
    """
    
    def __init__(self, config: P2HConfig):
        super().__init__()
        self.config = config
        
        # 拼音嵌入
        self.pinyin_embedding = nn.Embedding(
            config.pinyin_vocab_size, 
            config.d_model, 
            padding_idx=config.pad_id
        )
        
        # 汉字嵌入
        self.hanzi_embedding = nn.Embedding(
            config.hanzi_vocab_size,
            config.d_model,
            padding_idx=config.pad_id
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            config.d_model, 
            max(config.max_pinyin_len, config.max_hanzi_len),
            config.dropout
        )
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_encoder_layers,
            num_decoder_layers=config.n_decoder_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_projection = nn.Linear(config.d_model, config.hanzi_vocab_size)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        pinyin_ids: torch.Tensor,      # [batch, src_len]
        hanzi_ids: torch.Tensor,       # [batch, tgt_len]
        pinyin_mask: torch.Tensor = None,
        hanzi_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        前向传播（训练时使用）
        
        Returns:
            logits: [batch, tgt_len, hanzi_vocab_size]
        """
        # 嵌入
        src = self.pinyin_embedding(pinyin_ids)  # [batch, src_len, d_model]
        tgt = self.hanzi_embedding(hanzi_ids)    # [batch, tgt_len, d_model]
        
        # 位置编码
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # 创建掩码
        tgt_len = hanzi_ids.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(hanzi_ids.device)
        
        # padding mask
        src_key_padding_mask = (pinyin_ids == self.config.pad_id) if pinyin_mask is None else ~pinyin_mask
        tgt_key_padding_mask = (hanzi_ids == self.config.pad_id) if hanzi_mask is None else ~hanzi_mask
        
        # Transformer
        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # 输出投影
        logits = self.output_projection(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        pinyin_ids: torch.Tensor,      # [batch, src_len]
        max_len: int = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        生成汉字序列（推理时使用）
        
        Args:
            pinyin_ids: 拼音 ID 序列
            max_len: 最大生成长度
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
        Returns:
            generated: [batch, gen_len] 生成的汉字 ID
        """
        self.eval()
        device = pinyin_ids.device
        batch_size = pinyin_ids.size(0)
        max_len = max_len or self.config.max_hanzi_len
        
        # 编码器输出
        src = self.pinyin_embedding(pinyin_ids)
        src = self.pos_encoder(src)
        
        src_key_padding_mask = (pinyin_ids == self.config.pad_id)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # 初始化解码器输入 (BOS)
        generated = torch.full((batch_size, 1), self.config.bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # 解码器
            tgt = self.hanzi_embedding(generated)
            tgt = self.pos_encoder(tgt)
            
            tgt_len = generated.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
            
            output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
            
            # 获取最后一个位置的 logits
            logits = self.output_projection(output[:, -1, :])  # [batch, vocab]
            
            # 应用温度
            logits = logits / temperature
            
            # Top-K 采样
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-P 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否全部生成 EOS
            if (next_token == self.config.eos_id).all():
                break
        
        return generated
    
    @torch.no_grad()
    def beam_search(
        self,
        pinyin_ids: torch.Tensor,      # [1, src_len] 单个样本
        beam_size: int = 5,
        max_len: int = None,
        length_penalty: float = 0.6,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Beam Search 解码
        
        Returns:
            List of (token_ids, score) 按分数降序
        """
        self.eval()
        device = pinyin_ids.device
        max_len = max_len or self.config.max_hanzi_len
        
        # 编码
        src = self.pinyin_embedding(pinyin_ids)
        src = self.pos_encoder(src)
        src_key_padding_mask = (pinyin_ids == self.config.pad_id)
        memory = self.transformer.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # 扩展 memory 到 beam_size
        memory = memory.repeat(beam_size, 1, 1)
        
        # 初始化 beam
        # (token_ids, log_prob)
        beams = [(torch.tensor([[self.config.bos_id]], device=device), 0.0)]
        completed = []
        
        for step in range(max_len - 1):
            all_candidates = []
            
            for seq, score in beams:
                if seq[0, -1].item() == self.config.eos_id:
                    completed.append((seq, score))
                    continue
                
                # 解码
                tgt = self.hanzi_embedding(seq)
                tgt = self.pos_encoder(tgt)
                tgt_mask = self._generate_square_subsequent_mask(seq.size(1)).to(device)
                
                output = self.transformer.decoder(tgt, memory[:1], tgt_mask=tgt_mask)
                logits = self.output_projection(output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Top-K 扩展
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    token_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0)
                    token_log_prob = topk_log_probs[0, i].item()
                    
                    new_seq = torch.cat([seq, token_id], dim=1)
                    new_score = score + token_log_prob
                    
                    all_candidates.append((new_seq, new_score))
            
            if not all_candidates:
                break
            
            # 选择 top-k
            all_candidates.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
            beams = all_candidates[:beam_size]
        
        # 合并结果
        completed.extend(beams)
        completed.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
        
        return completed[:beam_size]


class P2HVocab:
    """P2H 词表管理"""
    
    def __init__(self):
        self.pinyin2id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.id2pinyin = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
        
        self.hanzi2id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.id2hanzi = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
    
    def build_from_dict(self, char_dict: dict, word_freq: dict = None, max_hanzi: int = 8000):
        """从词典构建词表"""
        # 构建拼音词表
        for pinyin in char_dict.keys():
            if pinyin not in self.pinyin2id:
                idx = len(self.pinyin2id)
                self.pinyin2id[pinyin] = idx
                self.id2pinyin[idx] = pinyin
        
        # 构建汉字词表（按频率排序）
        if word_freq:
            sorted_chars = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        else:
            # 从 char_dict 收集所有汉字
            all_chars = set()
            for chars in char_dict.values():
                all_chars.update(chars)
            sorted_chars = [(c, 0) for c in all_chars]
        
        for char, freq in sorted_chars[:max_hanzi - 4]:  # 保留特殊 token
            if char not in self.hanzi2id:
                idx = len(self.hanzi2id)
                self.hanzi2id[char] = idx
                self.id2hanzi[idx] = char
    
    def encode_pinyin(self, pinyins: List[str]) -> List[int]:
        """拼音序列 -> ID 序列"""
        return [self.pinyin2id.get(py, self.pinyin2id['<unk>']) for py in pinyins]
    
    def encode_hanzi(self, text: str) -> List[int]:
        """汉字序列 -> ID 序列"""
        return [self.hanzi2id.get(c, self.hanzi2id['<unk>']) for c in text]
    
    def decode_hanzi(self, ids: List[int]) -> str:
        """ID 序列 -> 汉字序列"""
        chars = []
        for id in ids:
            if id in (0, 1):  # pad, bos
                continue
            if id == 2:  # eos
                break
            chars.append(self.id2hanzi.get(id, '?'))
        return ''.join(chars)
    
    @property
    def pinyin_vocab_size(self) -> int:
        return len(self.pinyin2id)
    
    @property
    def hanzi_vocab_size(self) -> int:
        return len(self.hanzi2id)
    
    def save(self, path: str):
        """保存词表"""
        import orjson
        data = {
            'pinyin2id': self.pinyin2id,
            'hanzi2id': self.hanzi2id,
        }
        with open(path, 'wb') as f:
            f.write(orjson.dumps(data))
    
    def load(self, path: str):
        """加载词表"""
        import orjson
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
        
        self.pinyin2id = data['pinyin2id']
        self.id2pinyin = {v: k for k, v in self.pinyin2id.items()}
        self.hanzi2id = data['hanzi2id']
        self.id2hanzi = {v: k for k, v in self.hanzi2id.items()}


if __name__ == '__main__':
    # 测试模型结构
    config = P2HConfig()
    model = P2HModel(config)
    
    print(f"P2H 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    pinyin_ids = torch.randint(4, config.pinyin_vocab_size, (batch_size, src_len))
    hanzi_ids = torch.randint(4, config.hanzi_vocab_size, (batch_size, tgt_len))
    
    logits = model(pinyin_ids, hanzi_ids)
    print(f"输入拼音: {pinyin_ids.shape}")
    print(f"输入汉字: {hanzi_ids.shape}")
    print(f"输出 logits: {logits.shape}")
    
    # 测试生成
    generated = model.generate(pinyin_ids[:1], max_len=10)
    print(f"生成结果: {generated.shape}")
