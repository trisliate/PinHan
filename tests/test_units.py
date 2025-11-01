"""
单元测试：测试核心模块的功能正确性。
"""
import sys
import unittest
from pathlib import Path
import torch
import orjson

sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess'))

from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
from pinyin_utils import (
    normalize_pinyin,
    validate_pinyin,
    validate_pinyin_sequence,
    extract_tone,
    tone_mark_to_number,
)


class TestVocab(unittest.TestCase):
    """测试Vocab类."""
    
    def setUp(self):
        self.tokens = ['a', 'e', 'i', 'o', 'u']
        self.vocab = Vocab(self.tokens)
    
    def test_vocab_size(self):
        """测试词表大小."""
        self.assertGreater(len(self.vocab), 0)
    
    def test_token_to_id(self):
        """测试token转ID."""
        for token in self.tokens:
            idx = self.vocab.token_to_idx(token)
            self.assertGreater(idx, -1)
    
    def test_id_to_token(self):
        """测试ID转token."""
        for i in range(len(self.vocab)):
            token = self.vocab.idx_to_token(i)
            self.assertIsNotNone(token)
    
    def test_encode_decode(self):
        """测试编码和解码."""
        tokens = ['a', 'e', 'i']
        encoded = self.vocab.encode(tokens, add_bos_eos=True)
        self.assertGreater(len(encoded), len(tokens))  # 有BOS和EOS
        
        decoded = self.vocab.decode(encoded, skip_specials=True)
        self.assertEqual(len(decoded), len(tokens))
    
    def test_special_tokens(self):
        """测试特殊token."""
        self.assertIn(self.vocab.pad_token, self.vocab.token_to_id)
        self.assertIn(self.vocab.bos_token, self.vocab.token_to_id)
        self.assertIn(self.vocab.eos_token, self.vocab.token_to_id)
        self.assertIn(self.vocab.unk_token, self.vocab.token_to_id)


class TestPinyinUtils(unittest.TestCase):
    """测试拼音工具函数."""
    
    def test_normalize_pinyin(self):
        """测试拼音规范化."""
        self.assertEqual(normalize_pinyin('MA1'), 'ma1')
        self.assertEqual(normalize_pinyin('  ma1  '), 'ma1')
        self.assertEqual(normalize_pinyin('mā'), 'ma1')
    
    def test_validate_pinyin(self):
        """测试拼音验证."""
        self.assertTrue(validate_pinyin('ma1'))
        self.assertTrue(validate_pinyin('bei3'))
        self.assertFalse(validate_pinyin(''))
        self.assertFalse(validate_pinyin('123'))
    
    def test_validate_pinyin_sequence(self):
        """测试拼音序列验证."""
        self.assertTrue(validate_pinyin_sequence('ma1 bei3 jing1'))
        self.assertFalse(validate_pinyin_sequence('ma1 123 jing1'))
    
    def test_extract_tone(self):
        """测试声调提取."""
        base, tone = extract_tone('ma1')
        self.assertEqual(base, 'ma')
        self.assertEqual(tone, '1')
        
        base, tone = extract_tone('bei')
        self.assertEqual(base, 'bei')
        self.assertIsNone(tone)
    
    def test_tone_mark_to_number(self):
        """测试声调标记转数字."""
        self.assertEqual(tone_mark_to_number('mā'), 'ma1')
        self.assertEqual(tone_mark_to_number('hǎo'), 'hao3')
        self.assertEqual(tone_mark_to_number('ma'), 'ma')


class TestSeq2SeqTransformer(unittest.TestCase):
    """测试Seq2SeqTransformer模型."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.model = Seq2SeqTransformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            d_model=256,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
        ).to(self.device)
    
    def test_model_initialization(self):
        """测试模型初始化."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.src_tok_emb)
        self.assertIsNotNone(self.model.tgt_tok_emb)
    
    def test_forward_pass(self):
        """测试前向传播."""
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        src = torch.randint(0, self.src_vocab_size, (src_len, batch_size))
        tgt = torch.randint(0, self.tgt_vocab_size, (tgt_len, batch_size))
        
        with torch.no_grad():
            output = self.model(src, tgt)
        
        # 输出形状: (tgt_len, batch_size, tgt_vocab_size)
        self.assertEqual(output.shape, (tgt_len, batch_size, self.tgt_vocab_size))
    
    def test_forward_with_masks(self):
        """测试带mask的前向传播."""
        batch_size = 2
        src_len = 10
        tgt_len = 8
        
        src = torch.randint(0, self.src_vocab_size, (src_len, batch_size))
        tgt = torch.randint(0, self.tgt_vocab_size, (tgt_len, batch_size))
        
        tgt_mask = generate_square_subsequent_mask(tgt_len).to(self.device).float()
        src_key_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
        tgt_key_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool)
        
        with torch.no_grad():
            output = self.model(
                src, tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask.float(),
                tgt_key_padding_mask=tgt_key_padding_mask.float(),
                memory_key_padding_mask=src_key_padding_mask.float(),
            )
        
        self.assertEqual(output.shape, (tgt_len, batch_size, self.tgt_vocab_size))
    
    def test_parameter_count(self):
        """测试参数数量."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertEqual(total_params, trainable_params)


class TestGenerateSquareSubsequentMask(unittest.TestCase):
    """测试因果mask生成."""
    
    def test_mask_shape(self):
        """测试mask形状."""
        sz = 5
        mask = generate_square_subsequent_mask(sz)
        self.assertEqual(mask.shape, (sz, sz))
    
    def test_mask_causality(self):
        """测试mask的因果性."""
        sz = 5
        mask = generate_square_subsequent_mask(sz)
        
        # 下三角应该全是0（允许注意），上三角应该全是-inf（屏蔽）
        for i in range(sz):
            for j in range(sz):
                if i >= j:
                    # 下三角（含对角线）应该是0
                    self.assertEqual(mask[i, j].item(), 0.0)
                else:
                    # 上三角应该是-inf
                    self.assertEqual(mask[i, j].item(), float('-inf'))


class TestDataLoading(unittest.TestCase):
    """测试数据加载."""
    
    def test_jsonl_loading(self):
        """测试JSONL数据加载."""
        # 假设存在test_mini.jsonl
        test_file = Path(__file__).parent.parent / 'data' / 'test_mini.jsonl'
        
        if not test_file.exists():
            self.skipTest(f"{test_file} 不存在")
        
        count = 0
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                    self.assertIn('pinyin', data)
                    self.assertIn('hanzi', data)
                    count += 1
                except Exception as e:
                    self.fail(f"无法解析JSONL行: {e}")
        
        self.assertGreater(count, 0)


if __name__ == '__main__':
    unittest.main()
