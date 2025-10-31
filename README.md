# PinHan Seq2Seq 训练说明（中文）

本仓库包含一个用于微调 Seq2Seq 模型（例如 T5）的训练脚本 `traning.py`，训练数据位于 `data/tran.jsonl`。

快速开始（Windows PowerShell）：

1) 使用系统 Python 创建虚拟环境并激活，然后安装依赖：

```powershell
# 使用系统 Python 创建虚拟环境并激活（PowerShell）
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

说明：你要求使用“系统环境”创建虚拟环境，上面的命令会基于系统的 Python 可执行文件创建一个本地虚拟环境。

2) 运行训练示例：

```powershell
python .\traning.py --model_name_or_path t5-small --output_dir outputs/pinhan --per_device_train_batch_size 8 --num_train_epochs 3
```

重要说明：
- 数据格式：`data/tran.jsonl` 每行为 JSON，必须包含 `hanzi`（源）和 `pinyin`（目标）两个字段。
- 日志：训练日志会输出到控制台，同时保存到 `outputs/<output_dir>/train.log`，训练结束后保存 `train_metrics.json`。
- 如果 tokenizer 没有 pad token，脚本会自动添加 `<pad>` 并 resize 模型 embedding。

关于 RTX 3060 6GB：
- 显存 6GB 对于大型模型或较大 batch 很可能不足；建议使用较小模型（例如 `t5-small`）并把 `--per_device_train_batch_size` 调小（例如 4 或 2）。
- 如果显存不足，训练会报 CUDA OOM，这时请减小 batch 或使用更短的 `--max_source_length` / `--max_target_length`。

估算与建议：
- 脚本在训练前会给出一个基于样本数和 batch 的粗略时间估算（仅供参考），并在训练过程中记录真实耗时（每个 epoch 与总耗时）。
- 对于一个较简单的汉字->拼音映射任务，若数据量很小（如 500 行），在 t5-small 上通常几轮（3~10 轮）就能获得较好效果；但精确轮数依赖于数据质量、长度与噪声。

如果你希望我继续：
- 我可以把评估指标扩展为 ROUGE 或 BLEU；
- 或添加 `accelerate` 的配置说明以便在多卡/多机上运行；
- 或在训练结束时自动上传模型到 Hugging Face Hub（需要你的 token）。

-----

如何使用系统已安装的 GPU 库（例如全局安装的 torch-cu）

如果你的显卡驱动和 CUDA 版本、以及对应的 GPU 版 PyTorch 已经安装在系统（全局）环境，并且你希望在虚拟环境中直接复用它（避免重新安装大包），可以在创建虚拟环境时启用 system site-packages，这样虚拟环境可以访问全局安装的包。

注意事项：
- 使用全局包可能会导致包版本/依赖冲突；建议只在你确定全局 torch 可用且与你要安装的其它依赖兼容时使用。
- 如果以后需要完全隔离环境，不要使用此方式。

PowerShell 示例（基于系统 Python 创建可访问全局包的虚拟环境）：

```powershell
# 在仓库根目录创建并激活虚拟环境，允许访问系统 site-packages
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1

# 检查全局 torch 是否可用及是否识别 GPU
python -c "import torch; print('torch', getattr(torch,'__version__',None)); print('cuda_available', torch.cuda.is_available()); import sys; print('python', sys.executable)"
```

如果上面命令显示 GPU 可用（cuda_available True）并且 torch 版本正确，你可以跳过在虚拟环境中重新安装 torch。下面展示一种在 Windows PowerShell 中安装 requirements 时跳过 `torch` 的方法：

```powershell
# 生成一个临时 requirements 文件，排除以 'torch' 开头的行
$req = Get-Content requirements.txt | Where-Object { $_ -notmatch '^\s*torch' }
$req | Set-Content .\temp_requirements.txt

# 安装除 torch 以外的依赖
pip install -r .\temp_requirements.txt

# 清理临时文件
Remove-Item .\temp_requirements.txt
```

如果你更希望直接安装全部依赖并覆盖全局 torch（不推荐当你已有可用 GPU 版 torch 时），可以直接运行：

```powershell
pip install -r requirements.txt
```

但要注意：安装会尝试获取与系统兼容的 torch wheel，可能耗时较长并占用磁盘空间。

最后，做一次快速试跑来确认一切正常：

```powershell
python .\traning.py --model_name_or_path t5-small --output_dir outputs/pinhan_test --per_device_train_batch_size 2 --num_train_epochs 1 --eval_split_ratio 0.02
```

该试跑会打印训练前的时间估算，并在训练过程中记录真实每轮耗时（见 `outputs/pinhan_test/train.log`）。


