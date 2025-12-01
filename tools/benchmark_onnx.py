import os
import sys
import time
import torch
import numpy as np
import onnxruntime as ort
import logging

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2h.model import P2HModel, P2HConfig, P2HVocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

class ONNXModel:
    """ONNX 模型推理包装器"""
    def __init__(self, encoder_path, decoder_path, vocab: P2HVocab):
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
        self.vocab = vocab
        self.bos_id = 1
        self.eos_id = 2
        
    def beam_search(self, pinyin_ids, beam_size=5, max_len=20):
        # Pad pinyin_ids to 32
        padded_pinyin = np.zeros((1, 32), dtype=np.int64)
        length = min(len(pinyin_ids), 32)
        padded_pinyin[0, :length] = pinyin_ids[:length]
        
        # 1. Encoder
        memory = self.encoder_sess.run(['memory'], {'pinyin_ids': padded_pinyin})[0]
        
        # 2. Decoder Beam Search
        # beams: List of (seq, score)
        beams = [([self.bos_id], 0.0)]
        completed = []
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == self.eos_id:
                    completed.append((seq, score))
                    continue
                
                # Prepare input (Pad to 32)
                padded_tgt = np.zeros((1, 32), dtype=np.int64)
                curr_len = min(len(seq), 32)
                padded_tgt[0, :curr_len] = seq[:curr_len]
                
                # Run Decoder
                logits = self.decoder_sess.run(['logits'], {
                    'tgt_ids': padded_tgt,
                    'memory': memory
                })[0]
                
                # Get last token logits (at curr_len - 1)
                last_logits = logits[0, curr_len - 1, :]
                
                # Softmax (optional for ranking, but good for consistency)
                # simple log_softmax
                exp_logits = np.exp(last_logits - np.max(last_logits))
                log_probs = np.log(exp_logits / np.sum(exp_logits))
                
                # Top-K
                top_k_indices = np.argsort(log_probs)[-beam_size:][::-1]
                
                for idx in top_k_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + log_probs[idx]
                    candidates.append((new_seq, new_score))
            
            if not candidates:
                break
                
            # Select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # Early stop if all beams are completed (simplified)
            if all(b[0][-1] == self.eos_id for b in beams):
                completed.extend(beams)
                break
        
        # Final sort
        completed.extend([b for b in beams if b[0][-1] != self.eos_id])
        completed.sort(key=lambda x: x[1], reverse=True)
        
        return completed[:beam_size]

def benchmark():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, "models", "onnx")
    
    # 1. Load PyTorch Model
    logger.info("加载 PyTorch 模型...")
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'p2h', 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pt_model = P2HModel(checkpoint['config'])
    pt_model.load_state_dict(checkpoint['model_state_dict'])
    pt_model.eval()
    
    # Load Vocab
    vocab = P2HVocab()
    vocab.load(os.path.join(project_root, 'checkpoints', 'p2h', 'vocab.json'))
    
    # 2. Load ONNX Models
    logger.info("加载 ONNX 模型 (FP32)...")
    onnx_model = ONNXModel(
        os.path.join(model_dir, "p2h_encoder.onnx"),
        os.path.join(model_dir, "p2h_decoder.onnx"),
        vocab
    )
    
    logger.info("加载 ONNX 模型 (INT8)...")
    onnx_quant_model = ONNXModel(
        os.path.join(model_dir, "p2h_encoder_quant.onnx"),
        os.path.join(model_dir, "p2h_decoder_quant.onnx"),
        vocab
    )
    
    # Test Data
    test_pinyin = ["jin", "tian", "tian", "qi", "hen", "hao"]
    pinyin_ids = vocab.encode_pinyin(test_pinyin)
    pinyin_tensor = torch.tensor([pinyin_ids], dtype=torch.long)
    
    logger.info(f"测试输入: {' '.join(test_pinyin)}")
    
    # Warmup
    logger.info("预热中...")
    pt_model.beam_search(pinyin_tensor, beam_size=5)
    onnx_model.beam_search(pinyin_ids, beam_size=5)
    onnx_quant_model.beam_search(pinyin_ids, beam_size=5)
    
    # Benchmark Loop
    num_runs = 50
    
    # PyTorch
    start = time.perf_counter()
    for _ in range(num_runs):
        pt_model.beam_search(pinyin_tensor, beam_size=5)
    pt_time = (time.perf_counter() - start) / num_runs * 1000
    
    # ONNX FP32
    start = time.perf_counter()
    for _ in range(num_runs):
        onnx_model.beam_search(pinyin_ids, beam_size=5)
    onnx_time = (time.perf_counter() - start) / num_runs * 1000
    
    # ONNX INT8
    start = time.perf_counter()
    for _ in range(num_runs):
        onnx_quant_model.beam_search(pinyin_ids, beam_size=5)
    quant_time = (time.perf_counter() - start) / num_runs * 1000
    
    print("\n" + "="*50)
    print("性能测试结果 (Beam Size=5, Length=6)")
    print("="*50)
    print(f"PyTorch (CPU):  {pt_time:.2f} ms")
    print(f"ONNX (FP32):    {onnx_time:.2f} ms (x{pt_time/onnx_time:.2f})")
    print(f"ONNX (INT8):    {quant_time:.2f} ms (x{pt_time/quant_time:.2f})")
    print("="*50)
    
    # Verify Output
    print("\n输出验证:")
    pt_out = pt_model.beam_search(pinyin_tensor, beam_size=1)[0][0]
    onnx_out = onnx_model.beam_search(pinyin_ids, beam_size=1)[0][0]
    quant_out = onnx_quant_model.beam_search(pinyin_ids, beam_size=1)[0][0]
    
    # pt_out is [1, seq_len], flatten it
    print(f"PyTorch: {vocab.decode_hanzi(pt_out[0].tolist())}")
    print(f"ONNX:    {vocab.decode_hanzi(onnx_out)}")
    print(f"Quant:   {vocab.decode_hanzi(quant_out)}")

if __name__ == "__main__":
    benchmark()
