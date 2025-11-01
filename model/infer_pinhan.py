"""推理脚本：加载模型并进行拼音到汉字的转换."""
import sys
import time
import argparse
from pathlib import Path
import orjson
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from seq2seq_transformer import Seq2SeqTransformer, Vocab, generate_square_subsequent_mask

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(
    model_path: Path,
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_idx_src: int,
    pad_idx_tgt: int,
    device: torch.device = None,
) -> Seq2SeqTransformer:
    """加载模型."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Seq2SeqTransformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        pad_idx_src=pad_idx_src,
        pad_idx_tgt=pad_idx_tgt,
    )
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def greedy_decode(
    model: Seq2SeqTransformer,
    src_ids: list,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device = None,
    max_len: int = 64,
) -> list:
    """贪心解码."""
    if device is None:
        device = next(model.parameters()).device
    
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    src_key_padding_mask = (src_tensor.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
    with torch.no_grad():
        ys = torch.LongTensor([tgt_vocab.token_to_id[tgt_vocab.bos_token]]).unsqueeze(1).to(device)
        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(device)
            out = model(
                src_tensor,
                ys,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            prob = F.softmax(out[-1, 0], dim=-1)
            next_id = int(torch.argmax(prob).item())
            ys = torch.cat([ys, torch.LongTensor([[next_id]]).to(device)], dim=0)
            if next_id == tgt_vocab.token_to_id[tgt_vocab.eos_token]:
                break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ys.squeeze(1).tolist()


def beam_search_decode(
    model: Seq2SeqTransformer,
    src_ids: list,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    beam_size: int = 5,
    device: torch.device = None,
    max_len: int = 64,
) -> list[tuple]:
    """Beam Search 解码，返回 top-k 候选."""
    if device is None:
        device = next(model.parameters()).device
    
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)
    src_key_padding_mask = (src_tensor.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
    with torch.no_grad():
        ys = torch.LongTensor([tgt_vocab.token_to_id[tgt_vocab.bos_token]]).unsqueeze(1).to(device)
        candidates = [(ys, 0.0)]
        for step in range(max_len):
            new_candidates = []
            for seq, score in candidates:
                if seq[-1, 0].item() == tgt_vocab.token_to_id[tgt_vocab.eos_token]:
                    new_candidates.append((seq, score))
                    continue
                tgt_mask = generate_square_subsequent_mask(seq.size(0)).to(device)
                out = model(
                    src_tensor,
                    seq,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                prob = F.softmax(out[-1, 0], dim=-1)
                log_prob = torch.log(prob + 1e-10)
                top_k_prob, top_k_ids = torch.topk(log_prob, k=min(beam_size, len(tgt_vocab)))
                for prob_val, token_id in zip(top_k_prob, top_k_ids):
                    new_seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=0)
                    new_score = score + prob_val.item()
                    new_candidates.append((new_seq, new_score))
            new_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = new_candidates[:beam_size]
            if all(seq[-1, 0].item() == tgt_vocab.token_to_id[tgt_vocab.eos_token] for seq, _ in candidates):
                break
        results = []
        for seq, score in candidates[:5]:
            token_ids = seq.squeeze(1).tolist()
            results.append((token_ids, score))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    """主推理函数."""
    parser = argparse.ArgumentParser(description='拼音到汉字推理')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径 (e.g., outputs/5k_model/best_model.pt)')
    parser.add_argument('--pinyin', type=str, required=True, help='输入拼音 (e.g., "ni3 hao3")')
    parser.add_argument('--beam-size', type=int, default=1, help='束搜索大小 (默认1=贪心)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='设备选择')
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 获取模型目录
    model_path = Path(args.model)
    model_dir = model_path.parent
    
    print("加载模型和词表...")
    if not model_path.exists():
        print(f"❌ 错误：模型文件不存在 {model_path}")
        return
    
    if not (model_dir / 'src_vocab.json').exists() or not (model_dir / 'tgt_vocab.json').exists():
        print(f"❌ 错误：词表文件不存在于 {model_dir}")
        return
    
    # 加载词表
    src_vocab = Vocab.load(str(model_dir / 'src_vocab.json'))
    tgt_vocab = Vocab.load(str(model_dir / 'tgt_vocab.json'))
    
    # 加载模型
    model = load_model(
        model_path,
        len(src_vocab),
        len(tgt_vocab),
        pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token],
        pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token],
        device=device,
    )
    print("✅ 模型加载成功\n")
    
    # 推理
    print(f"输入拼音: {args.pinyin}")
    src_tokens = args.pinyin.strip().split()
    src_ids = src_vocab.encode(src_tokens, add_bos_eos=True)
    
    if args.beam_size == 1:
        # 贪心解码
        print("使用贪心解码...")
        start_time = time.time()
        pred_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device, max_len=64)
        elapsed = time.time() - start_time
        pred_tokens = tgt_vocab.decode(pred_ids)
        pred_text = ''.join(pred_tokens)
        print(f"预测汉字: {pred_text}")
        print(f"推理耗时: {elapsed * 1000:.2f}ms\n")
    else:
        # 束搜索
        print(f"使用束搜索 (beam_size={args.beam_size})...")
        start_time = time.time()
        candidates = beam_search_decode(model, src_ids, src_vocab, tgt_vocab, beam_size=args.beam_size, device=device)
        elapsed = time.time() - start_time
        
        print("候选列表（排序by分数）:")
        for rank, (pred_ids, score) in enumerate(candidates, 1):
            pred_tokens = tgt_vocab.decode(pred_ids)
            pred_text = ''.join(pred_tokens)
            print(f"  {rank}. {pred_text} (分数: {score:.4f})")
        print(f"推理耗时: {elapsed * 1000:.2f}ms\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"推理出错: {e}")
        import traceback
        traceback.print_exc()
