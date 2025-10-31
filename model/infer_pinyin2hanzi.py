"""
推理脚本：加载训练好的小型 Transformer 模型，对拼音输入进行贪心解码并打印汉字结果。
用法： python model\infer_pinyin2hanzi.py
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from seq2seq_transformer import Seq2SeqTransformer, Vocab, generate_square_subsequent_mask


MODEL_DIR = Path('outputs/small_model')
DATA_PATH = Path('data/tran.jsonl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_vocab(path: Path):
    return Vocab.load(str(path))


def load_model(model_path: Path, src_vocab_size, tgt_vocab_size, pad_idx_src, pad_idx_tgt):
    # 注意：传入源/目标的 pad idx，确保 embedding 使用正确的 padding 索引
    model = Seq2SeqTransformer(src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, pad_idx_src=pad_idx_src, pad_idx_tgt=pad_idx_tgt)
    ckpt = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=64):
    # src_ids: list of ids WITHOUT BOS/EOS? 我们 expect encode 已经包含 BOS/EOS
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(DEVICE)  # (S, 1)
    src_key_padding_mask = (src_tensor.transpose(0,1) == src_vocab.token_to_id[src_vocab.pad_token])
    # encoder forward
    with torch.no_grad():
        # 准备初始 tgt 为 BOS
        ys = torch.LongTensor([tgt_vocab.token_to_id[tgt_vocab.bos_token]]).unsqueeze(1).to(DEVICE)  # (1,1)
        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(0)).to(DEVICE)
            out = model(src_tensor, ys, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=None, memory_key_padding_mask=src_key_padding_mask)
            prob = F.softmax(out[-1, 0], dim=-1)
            # 打印第一步的 top-k 概率，帮助排查模型为何直接产生 EOS
            if i == 0:
                topk = torch.topk(prob, k=10)
                topk_vals = topk.values.cpu().numpy().tolist()
                topk_idxs = topk.indices.cpu().numpy().tolist()
                topk_tokens = [tgt_vocab.idx_to_token(idx) for idx in topk_idxs]
                print('第一步 topk tokens:', list(zip(topk_tokens, topk_vals)))
                eos_id = tgt_vocab.token_to_id[tgt_vocab.eos_token]
                print('EOS id:', eos_id, 'EOS token:', tgt_vocab.eos_token, 'prob:', float(prob[eos_id].cpu().numpy()))
            next_id = int(torch.argmax(prob).item())
            # 安全策略：若第一步直接预测 EOS（导致结果为空），则使用第二高概率的 token 以获得可读输出
            if i == 0:
                eos_id = tgt_vocab.token_to_id[tgt_vocab.eos_token]
                if next_id == eos_id:
                    # 取 top-2 中第二个（若存在）
                    topk = torch.topk(prob, k=2)
                    cand_idxs = topk.indices.cpu().numpy().tolist()
                    if len(cand_idxs) > 1 and cand_idxs[0] == eos_id:
                        next_id = int(cand_idxs[1])
            ys = torch.cat([ys, torch.LongTensor([[next_id]]).to(DEVICE)], dim=0)
            if next_id == tgt_vocab.token_to_id[tgt_vocab.eos_token]:
                break
    return ys.squeeze(1).tolist()  # 返回 id 列表


def main():
    print('加载词表与模型...')
    src_vocab = Vocab.load(str(MODEL_DIR / 'src_vocab.json'))
    tgt_vocab = Vocab.load(str(MODEL_DIR / 'tgt_vocab.json'))
    model = load_model(MODEL_DIR / 'model.pt', len(src_vocab), len(tgt_vocab), pad_idx_src=src_vocab.token_to_id[src_vocab.pad_token], pad_idx_tgt=tgt_vocab.token_to_id[tgt_vocab.pad_token])
    print('加载完成，示例推理：')
    # 从数据文件中取几条样本
    examples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            examples.append((j.get('pinyin'), j.get('hanzi')))
    # 运行推理
    for pinyin, hanzi in examples[:10]:
         src_tokens = pinyin.strip().split()
         src_ids = src_vocab.encode(src_tokens, add_bos_eos=True)
         # 打印源 id 与对应 token（含特殊 token），便于排查编码是否正确
         src_id_tokens = [src_vocab.idx_to_token(i) for i in src_ids]
         print('src_ids:', src_ids)
         print('src_id_tokens:', src_id_tokens)
         pred_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=64)
         # 打印预测 id 列表及对应 token（包含特殊 token），以及 decode 后的文本
         print('pred_ids:', pred_ids)
         pred_id_tokens = [tgt_vocab.idx_to_token(i) for i in pred_ids]
         print('pred_id_tokens:', pred_id_tokens)
         pred_tokens = tgt_vocab.decode(pred_ids)
         pred_text = ''.join(pred_tokens)
         print('拼音:', pinyin)
         print('参考:', hanzi)
         print('预测:', pred_text)
         print('---')


if __name__ == '__main__':
    main()
