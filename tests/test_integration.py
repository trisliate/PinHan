"""
集成测试：测试完整的训练-推理流程。
"""
import sys
import unittest
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess'))

from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
import torch.nn as nn
import torch.optim as optim


class SimpleDataset(torch.utils.data.Dataset):
    """简单测试数据集."""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, src_vocab, tgt_vocab):
    """批处理函数."""
    srcs, tgts = zip(*batch)
    src_ids = [src_vocab.encode(list(s), add_bos_eos=True) for s in srcs]
    tgt_ids = [tgt_vocab.encode(list(t), add_bos_eos=True) for t in tgts]
    
    max_src = max(len(x) for x in src_ids)
    max_tgt = max(len(x) for x in tgt_ids)
    
    pad_id_src = src_vocab.token_to_id[src_vocab.pad_token]
    pad_id_tgt = tgt_vocab.token_to_id[tgt_vocab.pad_token]
    
    src_padded = [x + [pad_id_src] * (max_src - len(x)) for x in src_ids]
    tgt_padded = [x + [pad_id_tgt] * (max_tgt - len(x)) for x in tgt_ids]
    
    src_tensor = torch.LongTensor(src_padded).transpose(0, 1)
    tgt_tensor = torch.LongTensor(tgt_padded).transpose(0, 1)
    
    return src_tensor, tgt_tensor


class TestTrainingPipeline(unittest.TestCase):
    """测试完整的训练流程."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
        # 创建简单的训练数据
        self.samples = [
            (['m', 'a'], ['妈']),
            (['b', 'a'], ['爸']),
            (['d', 'i'], ['弟']),
            (['j', 'i', 'e'], ['姐']),
        ]
        
        # 构建词表
        src_tokens = set()
        tgt_tokens = set()
        for src, tgt in self.samples:
            src_tokens.update(src)
            tgt_tokens.update(tgt)
        
        self.src_vocab = Vocab(list(src_tokens))
        self.tgt_vocab = Vocab(list(tgt_tokens))
    
    def test_single_epoch_training(self):
        """测试单轮训练."""
        dataset = SimpleDataset(self.samples)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.src_vocab, self.tgt_vocab),
        )
        
        model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=64,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tgt_vocab.token_to_id[self.tgt_vocab.pad_token]
        )
        
        model.train()
        total_loss = 0.0
        
        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
            src_key_padding_mask = (
                src.transpose(0, 1) == self.src_vocab.token_to_id[self.src_vocab.pad_token]
            )
            tgt_key_padding_mask = (
                tgt_input.transpose(0, 1) == self.tgt_vocab.token_to_id[self.tgt_vocab.pad_token]
            )
            
            # 转换为float
            tgt_mask = tgt_mask.float()
            src_key_padding_mask = src_key_padding_mask.float()
            tgt_key_padding_mask = tgt_key_padding_mask.float()
            
            optimizer.zero_grad()
            
            logits = model(
                src, tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.assertGreater(avg_loss, 0)
    
    def test_multiple_epochs_training(self):
        """测试多轮训练."""
        dataset = SimpleDataset(self.samples)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.src_vocab, self.tgt_vocab),
        )
        
        model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=64,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tgt_vocab.token_to_id[self.tgt_vocab.pad_token]
        )
        
        losses = []
        
        for epoch in range(3):
            model.train()
            total_loss = 0.0
            
            for src, tgt in dataloader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                tgt_input = tgt[:-1, :]
                tgt_out = tgt[1:, :]
                
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
                src_key_padding_mask = (
                    src.transpose(0, 1) == self.src_vocab.token_to_id[self.src_vocab.pad_token]
                )
                tgt_key_padding_mask = (
                    tgt_input.transpose(0, 1) == self.tgt_vocab.token_to_id[self.tgt_vocab.pad_token]
                )
                
                tgt_mask = tgt_mask.float()
                src_key_padding_mask = src_key_padding_mask.float()
                tgt_key_padding_mask = tgt_key_padding_mask.float()
                
                optimizer.zero_grad()
                
                logits = model(
                    src, tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
        
        # 确保有损失记录
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(l > 0 for l in losses))
    
    def test_model_save_load(self):
        """测试模型保存和加载."""
        model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=64,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ).to(self.device)
        
        # 保存模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model.pt'
            torch.save(model.state_dict(), model_path)
            
            # 加载模型
            new_model = Seq2SeqTransformer(
                len(self.src_vocab),
                len(self.tgt_vocab),
                d_model=64,
                nhead=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
            ).to(self.device)
            new_model.load_state_dict(torch.load(model_path))
            
            self.assertTrue(model_path.exists())


class TestInferencePipeline(unittest.TestCase):
    """测试推理流程."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
        self.samples = [
            (['m', 'a'], ['妈']),
            (['b', 'a'], ['爸']),
        ]
        
        src_tokens = set()
        tgt_tokens = set()
        for src, tgt in self.samples:
            src_tokens.update(src)
            tgt_tokens.update(tgt)
        
        self.src_vocab = Vocab(list(src_tokens))
        self.tgt_vocab = Vocab(list(tgt_tokens))
    
    def test_greedy_decoding(self):
        """测试贪心解码."""
        model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=64,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
        ).to(self.device)
        
        model.eval()
        
        # 简单输入
        src_tokens = ['m', 'a']
        src_ids = self.src_vocab.encode(src_tokens, add_bos_eos=True)
        src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            ys = torch.LongTensor([self.tgt_vocab.token_to_id[self.tgt_vocab.bos_token]]).unsqueeze(1).to(self.device)
            
            for _ in range(10):
                tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(self.device).float()
                out = model(
                    src_tensor, ys,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=torch.zeros(1, len(src_ids), dtype=torch.float),
                )
                prob = torch.softmax(out[-1, 0], dim=-1)
                next_id = torch.argmax(prob).item()
                ys = torch.cat([ys, torch.LongTensor([[next_id]]).to(self.device)], dim=0)
                
                if next_id == self.tgt_vocab.token_to_id[self.tgt_vocab.eos_token]:
                    break
        
        self.assertGreater(ys.size(0), 1)


if __name__ == '__main__':
    unittest.main()
