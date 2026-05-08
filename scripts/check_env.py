"""环境验证脚本。运行：python scripts/check_env.py

.. note::
   导入顺序很重要：pyarrow（datasets/pandas 的依赖）与 torch 在 Windows 上存在
   C++ DLL 冲突（access violation）。必须先导入 datasets/pandas/pyarrow，再导入
   torch，否则 pyarrow 初始化时会崩溃。
"""
import os
import sys
from pathlib import Path
from importlib import import_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
# 环境检查脚本仅验证包可导入，无需连接 HuggingFace Hub
# datasets / huggingface_hub 在 import 阶段可能发起网络请求，导致超时卡死
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def check_python():
    """检查 Python 版本是否为 3.10.x"""
    v = sys.version_info
    assert v.major == 3 and v.minor == 10, f"需 Python 3.10.x，当前 {v.major}.{v.minor}"
    print(f"[PASS] Python {v.major}.{v.minor}.{v.micro}")


def check_imports_lightweight():
    """先导入不依赖 torch/CUDA 的轻量包。

    Windows 上 pyarrow（datasets/pandas 的依赖）与 torch 的 C++ 原生库存在
    DLL 冲突（access violation）。若 torch 先于 pyarrow 加载，pyarrow 初始化
    时会发生内存访问违例崩溃。因此必须先将 datasets/pandas/pyarrow 导入完毕，
    再加载 torch。
    """
    pkgs = [
        "numpy", "yaml", "tqdm", "rich",
        "PIL", "pypdfium2", "datasets",
    ]
    for pkg in pkgs:
        try:
            import_module(pkg)
            print(f"[PASS] {pkg}")
        except Exception as e:
            print(f"[FAIL] {pkg}: {e}")


def check_torch_and_ml():
    """在轻量包加载之后再导入 torch 及 ML 栈，避免 pyarrow DLL 冲突"""
    import torch  # noqa: F811  -- 延迟导入以避免 pyarrow DLL 冲突
    assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动与 PyTorch 安装"
    print(f"[PASS] CUDA {torch.version.cuda} 可用")
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"       显存: {gb:.1f} GB")

    ml_pkgs = ["transformers", "colpali_engine", "einops"]
    for pkg in ml_pkgs:
        try:
            import_module(pkg)
            print(f"[PASS] {pkg}")
        except Exception as e:
            print(f"[FAIL] {pkg}: {e}")


def check_hf_cache():
    """输出 HuggingFace 缓存路径配置"""
    print(f"[PASS] HF_HOME = {HF_HOME}")
    print(f"[PASS] HF_HUB_CACHE = {HF_HOME / 'hub'}")
    print(f"[PASS] HF_DATASETS_CACHE = {HF_HOME / 'datasets'}")


if __name__ == "__main__":
    print("=" * 50)
    print("ZeroShotVDR 环境检查")
    print("=" * 50)
    check_python()
    check_imports_lightweight()
    check_torch_and_ml()
    check_hf_cache()
    print("=" * 50)
    print("检查完成，全部通过。")