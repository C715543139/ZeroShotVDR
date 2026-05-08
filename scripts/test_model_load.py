"""ColPali-v1.3 模型加载验证脚本。
从项目根目录运行: python scripts/test_model_load.py
"""
import os
import sys
from pathlib import Path

# 设置项目内 HF 缓存
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# sitecustomize 在运行脚本时由 Python 自动加载（sys.path[0] == 脚本所在目录）
# 但确保项目根目录在 path 中以便加载
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("ColPali-v1.3 模型加载与基本功能验证")
print("=" * 60)

# 1. 验证 sitecustomize 补丁已应用
import sitecustomize  # noqa: F401

import transformers.integrations.peft as transformers_peft

patched_name = getattr(transformers_peft._convert_peft_config_moe, "__name__", "")
if patched_name == "_patched_convert_peft_config_moe":
    print("[PASS] sitecustomize PEFT MoE 补丁已应用")
else:
    print(f"[INFO] sitecustomize 补丁状态: {patched_name}")

# 2. 加载模型
import torch
from colpali_engine.models import ColPali

print("\n正在加载 ColPali-v1.3 模型...")
model = ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()
print(f"[PASS] 模型加载成功")
print(f"      设备: {model.device}")
print(f"      精度: {model.dtype}")
total_params = sum(p.numel() for p in model.parameters())
print(f"      参数: {total_params:,}")

# 3. 测试图像编码
from PIL import Image
from transformers import AutoProcessor

# 3. 加载处理器 — 使用 ColPali 专用处理器。
print("\n加载处理器 (Processor)...")
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

processor = ColPaliProcessor.from_pretrained(
    "vidore/colpaligemma-3b-pt-448-base"
)
print(f"[PASS] 处理器加载成功: {type(processor).__name__}")

print("\n测试图像编码...")
test_img = Image.new("RGB", (448, 448), color="gray")
batch = processor.process_images([test_img]).to(model.device)
print(f"      输入 keys: {list(batch.keys())}")
with torch.no_grad():
    emb = model.forward(**batch)
print(f"[PASS] 图像编码成功")
print(f"      输出形状: {emb.shape}")  # [1, n_patches, dim]

# 4. 测试文本编码（查询编码）
print("\n测试文本编码（查询编码）...")
query_batch = processor.process_queries(["What is shown in this document?"]).to(model.device)
print(f"      输入 keys: {list(query_batch.keys())}")
with torch.no_grad():
    query_emb = model.forward(**query_batch)
print(f"[PASS] 文本编码成功")
print(f"      输出形状: {query_emb.shape}")  # [1, n_tokens, dim]

# 5. 显存使用报告
print(f"\n当前 CUDA 显存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"当前 CUDA 显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n" + "=" * 60)
print("ColPali-v1.3 模型验证全部通过！")
print("=" * 60)
