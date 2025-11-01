"""
性能测试：测试模型推理速度、内存占用等性能指标。
"""
import sys
import unittest
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'model'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'preprocess'))

from seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask


class TestInferencePerformance(unittest.TestCase):
    """测试推理性能."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建简单的词表
        src_tokens = ['m', 'a', 'b', 'd', 'e', 'i', 'n', 'g']
        tgt_tokens = ['妈', '爸', '弟', '姐', '妹', '哥']
        
        self.src_vocab = Vocab(src_tokens)
        self.tgt_vocab = Vocab(tgt_tokens)
        
        # 创建模型
        self.model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        ).to(self.device)
        
        self.model.eval()
    
    def test_inference_latency(self):
        """测试单次推理延迟."""
        src_tokens = ['m', 'a']
        src_ids = self.src_vocab.encode(src_tokens, add_bos_eos=True)
        src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(self.device)
        
        # 预热
        with torch.no_grad():
            ys = torch.LongTensor([self.tgt_vocab.token_to_id[self.tgt_vocab.bos_token]]).unsqueeze(1).to(self.device)
            _ = self.model(src_tensor, ys)
        
        # 测量
        num_runs = 10
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                ys = torch.LongTensor([self.tgt_vocab.token_to_id[self.tgt_vocab.bos_token]]).unsqueeze(1).to(self.device)
                
                start = time.time()
                _ = self.model(src_tensor, ys)
                elapsed = time.time() - start
                
                latencies.append(elapsed * 1000)  # 转为毫秒
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\n推理延迟统计:")
        print(f"  平均: {avg_latency:.2f}ms")
        print(f"  最小: {min_latency:.2f}ms")
        print(f"  最大: {max_latency:.2f}ms")
        
        # 推理延迟应该在合理范围内
        self.assertLess(avg_latency, 1000)  # 应该小于1秒
    
    def test_memory_usage(self):
        """测试内存占用."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            src_tokens = ['m', 'a']
            src_ids = self.src_vocab.encode(src_tokens, add_bos_eos=True)
            src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                ys = torch.LongTensor([self.tgt_vocab.token_to_id[self.tgt_vocab.bos_token]]).unsqueeze(1).to(self.device)
                _ = self.model(src_tensor, ys)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            print(f"\nGPU峰值内存: {peak_memory:.2f}MB")
        else:
            print("\n未使用GPU，跳过CUDA内存测试")
    
    def test_batch_inference_latency(self):
        """测试批量推理延迟."""
        batch_sizes = [1, 2, 4, 8]
        
        print(f"\n批量推理延迟:")
        
        for batch_size in batch_sizes:
            src_tokens = ['m', 'a']
            src_ids = self.src_vocab.encode(src_tokens, add_bos_eos=True)
            src_tensor = torch.LongTensor(src_ids).unsqueeze(1).repeat(1, batch_size).to(self.device)
            
            with torch.no_grad():
                ys = torch.LongTensor([self.tgt_vocab.token_to_id[self.tgt_vocab.bos_token]]).unsqueeze(1).repeat(1, batch_size).to(self.device)
                
                start = time.time()
                _ = self.model(src_tensor, ys)
                elapsed = time.time() - start
            
            latency_per_sample = elapsed * 1000 / batch_size  # 每样本毫秒
            print(f"  批大小{batch_size}: {elapsed*1000:.2f}ms ({latency_per_sample:.2f}ms/样本)")


class TestTrainingPerformance(unittest.TestCase):
    """测试训练性能."""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        src_tokens = ['m', 'a', 'b', 'd', 'e', 'i', 'n', 'g']
        tgt_tokens = ['妈', '爸', '弟', '姐', '妹', '哥']
        
        self.src_vocab = Vocab(src_tokens)
        self.tgt_vocab = Vocab(tgt_tokens)
        
        self.model = Seq2SeqTransformer(
            len(self.src_vocab),
            len(self.tgt_vocab),
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        ).to(self.device)
    
    def test_training_throughput(self):
        """测试训练吞吐量 (样本/秒)."""
        import torch.nn as nn
        import torch.optim as optim
        
        model = self.model
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tgt_vocab.token_to_id[self.tgt_vocab.pad_token]
        )
        
        batch_sizes = [4, 8, 16]
        
        print(f"\n训练吞吐量:")
        
        for batch_size in batch_sizes:
            # 创建模拟数据
            src_len = 10
            tgt_len = 8
            
            src = torch.randint(0, len(self.src_vocab), (src_len, batch_size)).to(self.device)
            tgt = torch.randint(0, len(self.tgt_vocab), (tgt_len, batch_size)).to(self.device)
            
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(self.device).float()
            
            # 计时
            start = time.time()
            num_steps = 10
            
            for _ in range(num_steps):
                optimizer.zero_grad()
                
                logits = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss.backward()
                optimizer.step()
            
            elapsed = time.time() - start
            throughput = (batch_size * num_steps) / elapsed
            time_per_step = elapsed / num_steps * 1000
            
            print(f"  批大小{batch_size}: {throughput:.1f} 样本/秒 ({time_per_step:.2f}ms/step)")


class TestModelSize(unittest.TestCase):
    """测试模型大小."""
    
    def test_parameter_count(self):
        """测试参数数量."""
        src_tokens = ['m', 'a', 'b', 'd', 'e', 'i', 'n', 'g']
        tgt_tokens = ['妈', '爸', '弟', '姐', '妹', '哥']
        
        src_vocab = Vocab(src_tokens)
        tgt_vocab = Vocab(tgt_tokens)
        
        model = Seq2SeqTransformer(
            len(src_vocab),
            len(tgt_vocab),
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / 1024 / 1024  # 假设float32
        
        print(f"\n模型大小:")
        print(f"  总参数: {total_params:,}")
        print(f"  估计大小: {model_size_mb:.2f}MB")
        
        self.assertGreater(total_params, 0)


if __name__ == '__main__':
    unittest.main()
