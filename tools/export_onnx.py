import os
import sys
import torch
import torch.nn as nn
import logging
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2h.model import P2HModel, P2HConfig
from config import EngineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ONNX_Export")

class EncoderWrapper(nn.Module):
    """
    P2H Encoder 包装器
    输入: pinyin_ids
    输出: memory (编码器输出)
    """
    def __init__(self, model: P2HModel):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, pinyin_ids):
        # 嵌入 + 位置编码
        src = self.model.pinyin_embedding(pinyin_ids)
        src = self.model.pos_encoder(src)
        
        # Padding Mask
        src_key_padding_mask = (pinyin_ids == self.config.pad_id)
        
        # Encoder
        memory = self.model.transformer.encoder(
            src, 
            src_key_padding_mask=src_key_padding_mask
        )
        return memory

class DecoderWrapper(nn.Module):
    """
    P2H Decoder 包装器
    输入: tgt_ids (当前生成的汉字序列), memory (编码器输出)
    输出: logits (下一个词的概率分布)
    """
    def __init__(self, model: P2HModel):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, tgt_ids, memory):
        # 嵌入 + 位置编码
        tgt = self.model.hanzi_embedding(tgt_ids)
        tgt = self.model.pos_encoder(tgt)
        
        # Causal Mask (防止看到未来)
        tgt_len = tgt.size(1)
        tgt_mask = self.model._generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Decoder
        output = self.model.transformer.decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask
        )
        
        # Projection
        logits = self.model.output_projection(output)
        return logits

def export_models(model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu') # 导出时使用 CPU 即可

    # 1. 加载模型
    logger.info("正在加载 PyTorch 模型...")
    checkpoint_path = os.path.join(model_dir, 'checkpoints', 'p2h', 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"找不到模型文件: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = P2HModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 导出 Encoder
    logger.info("正在导出 Encoder (Fixed Length=32)...")
    encoder = EncoderWrapper(model)
    # 使用固定长度 32
    FIXED_LEN = 32
    dummy_pinyin = torch.randint(1, 100, (1, FIXED_LEN), dtype=torch.long)
    
    encoder_out_path = os.path.join(output_dir, "p2h_encoder.onnx")
    torch.onnx.export(
        encoder,
        (dummy_pinyin,),
        encoder_out_path,
        input_names=['pinyin_ids'],
        output_names=['memory'],
        dynamic_axes={
            'pinyin_ids': {0: 'batch_size'}, # 只有 batch 动态
            'memory': {0: 'batch_size'}
        },
        opset_version=14
    )
    logger.info(f"Encoder 导出完成: {encoder_out_path}")

    # 3. 导出 Decoder
    logger.info("正在导出 Decoder (Fixed Length=32)...")
    decoder = DecoderWrapper(model)
    # Decoder 输入也固定为 32
    dummy_tgt = torch.randint(1, 100, (1, FIXED_LEN), dtype=torch.long)
    dummy_memory = encoder(dummy_pinyin) 
    
    decoder_out_path = os.path.join(output_dir, "p2h_decoder.onnx")
    torch.onnx.export(
        decoder,
        (dummy_tgt, dummy_memory),
        decoder_out_path,
        input_names=['tgt_ids', 'memory'],
        output_names=['logits'],
        dynamic_axes={
            'tgt_ids': {0: 'batch_size'},
            'memory': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14
    )
    logger.info(f"Decoder 导出完成: {decoder_out_path}")

    # 4. 量化 (Quantization)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        logger.info("正在进行 INT8 量化...")
        
        # 量化 Encoder
        quantize_dynamic(
            encoder_out_path,
            os.path.join(output_dir, "p2h_encoder_quant.onnx"),
            weight_type=QuantType.QUInt8
        )
        
        # 量化 Decoder
        quantize_dynamic(
            decoder_out_path,
            os.path.join(output_dir, "p2h_decoder_quant.onnx"),
            weight_type=QuantType.QUInt8
        )
        logger.info("量化完成！")
        
    except ImportError:
        logger.warning("未安装 onnxruntime，跳过量化步骤。请运行: pip install onnxruntime")
    except Exception as e:
        logger.error(f"量化失败: {e}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "models", "onnx")
    export_models(project_root, output_dir)
