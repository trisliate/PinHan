"""推理脚本：加载模型并进行拼音到汉字的转换."""
import sys
import time
from pathlib import Path
import orjson
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from seq2seq_transformer import Seq2SeqTransformer, Vocab, generate_square_subsequent_mask

MODEL_DIR = Path('outputs/pinhan_model')
DATA_PATH = Path('data/clean_wiki.jsonl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(
    model_path: Path,
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_idx_src: int,
    pad_idx_tgt: int,
) -> Seq2SeqTransformer:
    """加载模型."""
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
    ckpt = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def greedy_decode(
    model: Seq2SeqTransformer,
    src_ids: list,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_len: int = 64,
) -> list:
    """贪心解码."""
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(DEVICE)
    src_key_padding_mask = (src_tensor.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
    with torch.no_grad():
        ys = torch.LongTensor([tgt_vocab.token_to_id[tgt_vocab.bos_token]]).unsqueeze(1).to(DEVICE)
        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
            out = model(
                src_tensor,
                ys,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            prob = F.softmax(out[-1, 0], dim=-1)
            next_id = int(torch.argmax(prob).item())
            ys = torch.cat([ys, torch.LongTensor([[next_id]]).to(DEVICE)], dim=0)
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
    max_len: int = 64,
) -> list[tuple]:
    """Beam Search 解码，返回 top-k 候选."""
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(DEVICE)
    src_key_padding_mask = (src_tensor.transpose(0, 1) == src_vocab.token_to_id[src_vocab.pad_token])
    with torch.no_grad():
        ys = torch.LongTensor([tgt_vocab.token_to_id[tgt_vocab.bos_token]]).unsqueeze(1).to(DEVICE)
        candidates = [(ys, 0.0)]
        for step in range(max_len):
            new_candidates = []
            for seq, score in candidates:
                if seq[-1, 0].item() == tgt_vocab.token_to_id[tgt_vocab.eos_token]:
                    new_candidates.append((seq, score))
                    continue
                tgt_mask = generate_square_subsequent_mask(seq.size(0)).to(DEVICE)
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
                    new_seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(DEVICE)], dim=0)
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
    print("加载模型和词表...")
    if not MODEL_DIR.exists():
        print(f"错误：模型目录 {MODEL_DIR} 不存在")
        return
    src_vocab = Vocab.load(str(MODEL_DIR / 'src_vocab.json'))
    tgt_vocab = Vocab.load(str(MODEL_DIR / 'tgt_vocab.json'))
    model = load_model(
        MODEL_DIR / 'model.pt',
        len(src_vocab),
        len(tgt_vocab),
        pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token],
        pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token],
    )
    print("加载完成，示例推理：\n")
    examples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            line = line.strip()
            if not line:
                continue
            j = orjson.loads(line)
            examples.append((j.get('pinyin'), j.get('hanzi')))
    print("=== 贪心解码 ===")
    total_time = 0.0
    for i, (pinyin, hanzi) in enumerate(examples[:5]):
        src_tokens = pinyin.strip().split()
        src_ids = src_vocab.encode(src_tokens, add_bos_eos=True)
        start_time = time.time()
        pred_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=64)
        elapsed = time.time() - start_time
        total_time += elapsed
        pred_tokens = tgt_vocab.decode(pred_ids)
        pred_text = ''.join(pred_tokens)
        print(f"\n示例 {i + 1}:")
        print(f"  拼音: {pinyin}")
        print(f"  参考: {hanzi}")
        print(f"  预测: {pred_text}")
        print(f"  耗时: {elapsed * 1000:.2f}ms")
    print(f"\n平均推理时间: {total_time / 5 * 1000:.2f}ms")
    print("\n\n=== Beam Search 解码 ===")
    pinyin, hanzi = examples[0]
    src_tokens = pinyin.strip().split()
    src_ids = src_vocab.encode(src_tokens, add_bos_eos=True)
    print(f"拼音: {pinyin}")
    print(f"参考: {hanzi}\n候选列表（置信度）:")
    start_time = time.time()
    candidates = beam_search_decode(model, src_ids, src_vocab, tgt_vocab, beam_size=5)
    elapsed = time.time() - start_time
    for rank, (pred_ids, score) in enumerate(candidates, 1):
        pred_tokens = tgt_vocab.decode(pred_ids)
        pred_text = ''.join(pred_tokens)
        print(f"  {rank}. {pred_text} (分数: {score:.4f})")
    print(f"耗时: {elapsed * 1000:.2f}ms")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"推理出错: {e}")
        import traceback
        traceback.print_exc()
