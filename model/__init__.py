"""
PinHan 模型包：包含 Seq2Seq Transformer 模型、训练、推理、评估脚本。
"""

from .core import (
    Vocab,
    Seq2SeqTransformer,
    generate_square_subsequent_mask,
    TrainingCheckpointManager,
    resume_or_init,
    load_trained_model,
    normalize_pinyin,
    normalize_pinyin_sequence,
    normalize_light_tone,
    validate_pinyin,
    validate_pinyin_sequence,
    extract_tone,
    tone_mark_to_number,
)

__all__ = [
    'Vocab',
    'Seq2SeqTransformer',
    'generate_square_subsequent_mask',
    'TrainingCheckpointManager',
    'resume_or_init',
    'load_trained_model',
    'normalize_pinyin',
    'normalize_pinyin_sequence',
    'normalize_light_tone',
    'validate_pinyin',
    'validate_pinyin_sequence',
    'extract_tone',
    'tone_mark_to_number',
]
