# **经过验证的 LeRobot 0.5.2 和 Pi0.5 的安装方式**

> ⚠️ **如果你遇到了任何错误请及时私信**



租机配置

选项	必须选这个	（主要是因为我只试过这个）
显卡	RTX 5090	
镜像	PyTorch 2.8.0 / Python 3.12 (Ubuntu 22.04) / CUDA 12.8	
磁盘	≥ 50GB 数据盘	
计费	按量计费	

> 为什么必须 2.8.0：5090 是 Blackwell 架构（sm_120），PyTorch 2.5.x 不支持，运行时会报 `no kernel image`。

---

0：开启学术加速（对官方 huggingface.co 有效）
```bash
source /etc/network_turbo
```


一、永久环境变量（写入 `~/.bashrc`）

```bash
cat >> ~/.bashrc << 'EOF'

# ===== Pi0.5 部署环境 =====
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_cache
export HF_TOKEN=hf_你的Token
EOF

source ~/.bashrc
mkdir -p /root/autodl-tmp/hf_cache/hub /root/autodl-tmp/lerobot_cache
```

验证：

```bash
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
```

预期输出：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/root/autodl-tmp/hf_cache
```

---

二、验证 PyTorch（base 环境可以开箱即用）

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('可用:', torch.cuda.is_available())
print('BF16:', torch.cuda.is_bf16_supported())
x = torch.tensor([1.0]).cuda()
print('Tensor:', x * 2)
"
```

预期输出：

```text
PyTorch: 2.8.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090
可用: True
BF16: True
Tensor: tensor([2.], device='cuda:0')
```

---

三、安装 LeRobot

```bash
pip install -U pip setuptools wheel typing_extensions numpy packaging

cd /root/autodl-tmp
git clone https://github.com/huggingface/lerobot.git lerobot
cd lerobot
pip install -e ".[pi]"
```

验证：

```bash
python -c "
import lerobot
from lerobot.policies.pi05 import PI05Policy
print('LeRobot 版本:', lerobot.__version__)
print('✅ PI05Policy 导入成功')
"
```

预期输出：

```text
LeRobot 版本: 0.5.X（比较新的，我用的是0.5.2）
✅ PI05Policy 导入成功
```

> 如果报错 `ValueError: LEROBOT_HOME is deprecated`，检查是否用了 `HF_LEROBOT_HOME` 而非旧名。

---

四、下载 Pi0.5 权重（hf-mirror 断点续传）

```bash
cd /root/autodl-tmp

python -c "
from huggingface_hub import snapshot_download
print('开始下载 lerobot/pi05_base（约14-15GB）...')
snapshot_download(
    repo_id='lerobot/pi05_base',
    local_dir='/root/autodl-tmp/pi05_base_fixed',
    allow_patterns=['*']
)
print('✅ 下载完成')
"
```

验证：

```bash
ls -lh /root/autodl-tmp/pi05_base_fixed/
ls -lh /root/autodl-tmp/pi05_base_fixed/*.safetensors
```

预期输出：

```text
total 14G
-rw-r--r-- 1 root root 5.0K ... README.md
-rw-r--r-- 1 root root 1.9K ... config.json
-rw-r--r-- 1 root root  14G ... model.safetensors
-rw-r--r-- 1 root root  450 ... policy_postprocessor.json
-rw-r--r-- 1 root root 1.1K ... policy_preprocessor.json
-rw-r--r-- 1 root root  14G /root/autodl-tmp/pi05_base_fixed/model.safetensors
```

> `model.safetensors` 必须约 14-15GB，如果只有几百 MB 说明下载中断，重新运行同一命令会自动续传

---

五、下载 Gated Tokenizer（必须走官方源 + Token）

关键说明：Pi0.5 内部依赖 `google/paligemma-3b-pt-224` tokenizer，这是 gated 模型。`hf-mirror.com` 无法代理权限验证，必须切回官方源 + 你的 HF Token

前置条件：去 https://huggingface.co/google/paligemma-3b-pt-224 点击 "Acknowledge license" 完成授权

```bash
# 临时取消镜像，走官方源
unset HF_ENDPOINT

python -c "
from transformers import AutoTokenizer
print('下载 google/paligemma-3b-pt-224 tokenizer...')
AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224', token=True)
print('✅ Tokenizer 缓存完成')
"

# 恢复镜像（后续如果要下其他权重还用得到）
export HF_ENDPOINT=https://hf-mirror.com
```

验证：

```bash
ls -la $HF_HOME/hub/models--google--paligemma-3b-pt-224/
```

预期：目录存在且包含 `snapshots/` 和 `refs/` 子目录

---

六、修复 config.json

```bash
python -c "
import json
from pathlib import Path
p = Path('/root/autodl-tmp/pi05_base_fixed/config.json')
if p.exists():
    cfg = json.load(p.open(encoding='utf-8'))
    removed = [k for k in ['use_peft', 'freeze_vision_encoder', 'train_expert_only'] if cfg.pop(k, None)]
    json.dump(cfg, p.open('w', encoding='utf-8'), indent=2)
    print(f'✅ 已处理: {removed if removed else \"无需修复\"}')
else:
    print('⚠️ config.json 不存在')
"
```

预期输出：

```text
✅ 已处理: 无需修复
# 或者
✅ 已处理: ['use_peft', ...]
```

---

七、推理验证（端到端打通）

```bash
cat << 'PYEOF' > /root/autodl-tmp/test_pi05.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import traceback
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

try:
    print("=== Pi0.5 base 加载测试 ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    local_path = "/root/autodl-tmp/pi05_base_fixed"

    policy = PI05Policy.from_pretrained(
        local_path,
        strict=False,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    preprocess, postprocess = make_pi05_pre_post_processors(policy.config)

    print(f"✅ 模型 + Processor 加载成功！显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    print("\n=== 测试推理 ===")
    dummy_obs = {
        "observation.images.base_0_rgb": torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32),
        "observation.images.left_wrist_0_rgb": torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32),
        "observation.images.right_wrist_0_rgb": torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32),
        "observation.state": torch.randn(1, 32, device="cuda", dtype=torch.float32),
        "task": "把红色的杯子放到桌子上",
    }

    batch = preprocess(dummy_obs)

    with torch.inference_mode():
        action_chunk = policy.select_action(batch)

    action = postprocess(action_chunk)

    print(f"✅ 推理成功！动作形状: {action.shape}")
    print(f"动作范围: [{action.min():.3f}, {action.max():.3f}]")
    print(f"最终显存: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

except Exception as e:
    print(f"❌ 报错: {e}")
    traceback.print_exc()
PYEOF

python /root/autodl-tmp/test_pi05.py
```

预期输出：

```text
=== Pi0.5 base 加载测试 ===
PyTorch: 2.8.0+cu128
GPU: NVIDIA GeForce RTX 5090
The PI05 model is a direct port of the OpenPI implementation...
WARNING:lerobot.configs.policies:Device 'mps' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'mps' is not available. Switching to 'cuda'.
Loading model from: /root/autodl-tmp/pi05_base_fixed
✓ Loaded state dict from model.safetensors
...
Remapped 812 state dict keys
All keys loaded successfully!
✅ 模型 + Processor 加载成功！显存: 15.5 GB

=== 测试推理 ===
✅ 推理成功！动作形状: torch.Size([1, 32])
动作范围: [-1.459, 1.028]
最终显存: 15.5 GB
```

---

常见错误速查表

`RuntimeError: Unable to read repodata JSON`	清华 Anaconda 源已废弃	

`ModuleNotFoundError: No module named 'torch'`	在新 conda 环境而非 base	使用自带 PyTorch 2.8.0 的镜像，base 环境直接可用

`ValueError: LEROBOT_HOME is deprecated`	旧环境变量名	改用 `HF_LEROBOT_HOME`	

`UnicodeEncodeError: 'ascii' codec...`	`HF_TOKEN` 包含中文或占位符，用`echo $HF_TOKEN`检查，确保是真实英文 Token	
