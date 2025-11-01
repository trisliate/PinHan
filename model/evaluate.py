"""评估脚本：计算模型准确率和其他指标."""
import sys
import orjson
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from seq2seq_transformer import Seq2SeqTransformer, Vocab, generate_square_subsequent_mask

# 常见多音字表（与 pinyin_utils.py 保持同步）
COMMON_POLYPHONIC_CHARS = {
    '中': ['zhong1', 'zhong4'],
    '长': ['chang2', 'zhang3'],
    '还': ['hai2', 'huan2'],
    '行': ['xing2', 'hang2'],
    '度': ['du4', 'duo2'],
    '重': ['zhong4', 'chong2'],
    '为': ['wei2', 'wei4'],
    '了': ['le', 'liao3'],
    '着': ['zhe', 'zhao2', 'zhu2'],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: Path, src_vocab: Vocab, tgt_vocab: Vocab) -> Seq2SeqTransformer:
    """加载模型."""
    model = Seq2SeqTransformer(
        len(src_vocab),
        len(tgt_vocab),
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token],
        pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token],
    )
    ckpt = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def greedy_decode(model: Seq2SeqTransformer, src_ids: list, src_vocab: Vocab, tgt_vocab: Vocab, max_len: int = 64) -> list:
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
    return ys.squeeze(1).tolist()


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算编辑距离（Levenshtein Distance）."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """计算评估指标."""
    assert len(predictions) == len(references), "预测和参考数量必须相同"
    
    metrics = {
        'total': len(predictions),
        'exact_match': 0,
        'avg_edit_distance': 0.0,
        'char_accuracy': 0.0,
        'polyphonic_accuracy': 0.0,
    }
    
    edit_distances = []
    char_correct = 0
    char_total = 0
    poly_correct = 0
    poly_total = 0
    polyphonic_chars = set(COMMON_POLYPHONIC_CHARS.keys())
    
    for pred, ref in zip(predictions, references):
        if pred == ref:
            metrics['exact_match'] += 1
        
        edit_dist = levenshtein_distance(pred, ref)
        edit_distances.append(edit_dist)
        
        # 字符级准确率
        for p, r in zip(pred, ref):
            char_total += 1
            if p == r:
                char_correct += 1
            if r in polyphonic_chars:
                poly_total += 1
                if p == r:
                    poly_correct += 1
    
    metrics['exact_match_rate'] = metrics['exact_match'] / len(predictions) * 100
    metrics['avg_edit_distance'] = sum(edit_distances) / len(edit_distances)
    metrics['char_accuracy'] = char_correct / char_total * 100 if char_total > 0 else 0.0
    metrics['polyphonic_accuracy'] = poly_correct / poly_total * 100 if poly_total > 0 else 0.0
    
    return metrics


def evaluate(data_path: Path, model_dir: Path, max_samples: int = 0) -> dict:
    """评估模型."""
    print(f"加载模型和词表...")
    src_vocab = Vocab.load(str(model_dir / 'src_vocab.json'))
    tgt_vocab = Vocab.load(str(model_dir / 'tgt_vocab.json'))
    model = load_model(model_dir / 'model.pt', src_vocab, tgt_vocab)
    
    print(f"加载数据...")
    predictions = []
    references = []
    count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples > 0 and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            
            try:
                j = orjson.loads(line)
            except Exception:
                continue
            
            pinyin = j.get('pinyin', '').strip()
            hanzi = j.get('hanzi', '').strip()
            
            if not pinyin or not hanzi:
                continue
            
            src_tokens = pinyin.split()
            src_ids = src_vocab.encode(src_tokens, add_bos_eos=True)
            
            pred_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=64)
            pred_tokens = tgt_vocab.decode(pred_ids)
            pred_text = ''.join(pred_tokens)
            
            predictions.append(pred_text)
            references.append(hanzi)
            count += 1
            
            if count % 100 == 0:
                print(f"已评估 {count} 个样本...")
    
    print(f"\n计算指标...")
    metrics = compute_metrics(predictions, references)
    
    return metrics, predictions, references


def main():
    """主评估函数."""
    import argparse
    parser = argparse.ArgumentParser(description='评估拼音->汉字模型')
    parser.add_argument('--data', type=str, default='data/clean_wiki.jsonl', help='测试数据路径')
    parser.add_argument('--model-dir', type=str, default='outputs/pinhan_model', help='模型目录')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大评估样本数（0 表示全部）')
    args = parser.parse_args()
    
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    
    if not data_path.exists():
        print(f"错误：数据文件不存在 {data_path}")
        sys.exit(1)
    
    if not model_dir.exists():
        print(f"错误：模型目录不存在 {model_dir}")
        sys.exit(1)
    
    metrics, predictions, references = evaluate(data_path, model_dir, args.max_samples)
    
    print("\n" + "="*50)
    print("📊 评估结果")
    print("="*50)
    print(f"总样本数: {metrics['total']}")
    print(f"完全匹配准确率: {metrics['exact_match_rate']:.2f}% ({metrics['exact_match']}/{metrics['total']})")
    print(f"字符级准确率: {metrics['char_accuracy']:.2f}%")
    print(f"多音字准确率: {metrics['polyphonic_accuracy']:.2f}%")
    print(f"平均编辑距离: {metrics['avg_edit_distance']:.2f}")
    print("="*50)
    
    # 保存结果
    results = {
        'metrics': metrics,
        'examples': [
            {'pinyin': ref, 'reference': ref, 'prediction': pred}
            for ref, pred in list(zip(references, predictions))[:20]
        ]
    }
    
    with open(model_dir / 'evaluation_results.json', 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    
    print(f"\n详细结果已保存到: {model_dir / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()
