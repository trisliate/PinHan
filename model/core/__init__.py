"""
PinHan 核心模块：Seq2Seq Transformer 模型、词表、检查点管理、拼音工具。
"""

from .seq2seq_transformer import Vocab, Seq2SeqTransformer, generate_square_subsequent_mask
from .checkpoint_manager import TrainingCheckpointManager, resume_or_init, load_trained_model
from .pinyin_utils import (
    normalize_pinyin,
    normalize_pinyin_sequence,
    normalize_light_tone,
    validate_pinyin,
    validate_pinyin_sequence,
    extract_tone,
    tone_mark_to_number,
    split_pinyin_sequence,
    join_pinyin_sequence,
    is_polyphonic_char,
    get_possible_pinyins,
    disambiguate_polyphonic,
    get_polyphonic_statistics,
    PinyinStatistics,
)

__all__ = [
    # Transformer 模型
    'Vocab',
    'Seq2SeqTransformer',
    'generate_square_subsequent_mask',
    # 检查点管理
    'TrainingCheckpointManager',
    'resume_or_init',
    'load_trained_model',
    # 拼音工具
    'normalize_pinyin',
    'normalize_pinyin_sequence',
    'normalize_light_tone',
    'validate_pinyin',
    'validate_pinyin_sequence',
    'extract_tone',
    'tone_mark_to_number',
    'split_pinyin_sequence',
    'join_pinyin_sequence',
    'is_polyphonic_char',
    'get_possible_pinyins',
    'disambiguate_polyphonic',
    'get_polyphonic_statistics',
    'PinyinStatistics',
]
