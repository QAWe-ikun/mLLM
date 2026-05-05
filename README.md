# 多模态具身VLA Agent - AI2-THOR

基于视觉-语言-动作(Vision-Language-Action)模型的具身智能导航Agent，在AI2-THOR仿真环境中实现多模态目标导航与交互任务。

## 项目简介

本项目构建一个"视觉+语言"导航Agent，给定自然语言指令与第一人称视觉观测，Agent规划并执行动作序列来完成目标导航(ObjectNav)和拾取交互(Pickup)任务。

**核心架构**：
- **仿真环境**: AI2-THOR (WSL/Linux headless模式)
- **视觉编码**: CLIP ViT-B/32 + MobileSAM
- **VLA基座**: Qwen3-VL 8B (QLoRA 4bit微调)
- **强化学习**: PPO (SFT + RL两阶段训练)
- **位置编码**: Agent 3D坐标+旋转角 → 可学习embedding

## 快速开始

### 环境要求

- Python 3.12 (推荐)
- GPU: RTX 4060 Ti (16GB) 或更高
- 操作系统: WSL2 或 Linux

### 1. 环境准备 (WSL/Linux)

```bash
# 使用 conda 创建环境
conda create -n mllm python=3.12
conda activate mllm
```

### 2. 安装 PyTorch (GPU版本)

```bash
# CUDA 12.x (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 3. 下载预训练模型 (使用 ModelScope 国内镜像)

模型默认从本地加载（不在线下载），使用脚本通过 ModelScope 下载：

```bash
# 下载全部模型 (Qwen3-VL 8B + CLIP)
python scripts/download_model.py

# 只下载 Qwen3-VL 8B
python scripts/download_model.py --only-qwen

# 只下载 CLIP ViT-B/32
python scripts/download_model.py --only-clip

# 强制重新下载
python scripts/download_model.py --force
```

如果模型不存在，代码会自动 fallback 到 dummy 模式进行测试。

### 4. 安装 MobileSAM (需从 GitHub 安装)

**方法A: 如果WSL能访问GitHub**
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm==0.6.7
```

**方法B: Windows下载 + WSL安装 (网络受限时)**
```bash
# Windows (PowerShell/CMD):
cd C:\\Users\\YourName\\Downloads
git clone https://github.com/ChaoningZhang/MobileSAM.git

# 回到WSL:
cd /mnt/c/Users/YourName/Downloads/MobileSAM
pip install -e .
pip install timm==0.6.7
```

### 4. 安装其余依赖

```bash
cd /path/to/mLLM
pip install -r requirements.txt
```

### 运行

```bash
# 1. 生成SFT演示数据
python scripts/01_generate_sft_data.py --num 5000

# 2. SFT阶段训练
python scripts/02_train_sft.py

# 3. PPO阶段训练
python scripts/03_train_ppo.py

# 4. 评估
python scripts/04_evaluate.py

# 5. 可视化
python scripts/05_visualize.py
```

## 项目结构

```
├── configs/                    # YAML配置文件
│   ├── env_config.yaml         # AI2-THOR环境配置
│   ├── model_config.yaml       # 模型架构+QLoRA配置
│   └── train_config.yaml       # 训练超参数
├── src/
│   ├── environment/            # 环境模块
│   │   ├── ai2thor_wrapper.py  # AI2-THOR环境封装
│   │   └── tasks/              # 任务定义
│   ├── perception/             # 感知模块
│   │   ├── clip_encoder.py     # CLIP特征提取
│   │   ├── mobile_sam.py       # MobileSAM目标检测
│   │   └── position_encoder.py # 位置编码器
│   │   └── feature_fusion.py   # 多模态特征融合
│   ├── models/                 # 模型模块
│   │   ├── vla_backbone.py     # Qwen3-VL QLoRA封装
│   │   └── action_head.py      # 动作输出头
│   ├── agent/                  # Agent模块
│   │   ├── ppo_trainer.py      # PPO训练器
│   │   ├── sft_trainer.py      # SFT训练器
│   │   └── rollout_buffer.py   # 经验回放
│   └── evaluation/             # 评估模块
│       ├── metrics.py          # SR/SPL计算
│       ├── eval_runner.py      # 评估运行器
│       └── visualization.py    # 轨迹可视化
├── scripts/                    # 运行脚本
│   ├── 01_generate_sft_data.py
│   ├── 02_train_sft.py
│   ├── 03_train_ppo.py
│   ├── 04_evaluate.py
│   └── 05_visualize.py
├── docs/                       # 文档
│   ├── 开题报告.md
│   └── 开题展示.pptx
└── results/                    # 实验结果输出
    ├── metrics/
    ├── trajectories/
    └── failure_analysis/
```

## 技术细节

### 模型配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen3-VL 8B-Instruct |
| 量化 | 4bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| 可训练参数 | ~7M (0.1%) |

### 动作空间

6个离散动作：
- `MoveAhead`: 向前移动0.25m
- `RotateLeft`: 向左旋转30°
- `RotateRight`: 向右旋转30°
- `LookUp`: 向上看30°
- `LookDown`: 向下看30°
- `Pickup`: 拾取目标物体

### 奖励函数

| 事件 | 奖励 |
|------|------|
| 到达目标 | +10.0 |
| 成功拾取 | +5.0 |
| 每步惩罚 | -0.1 |
| 靠近目标 | +0.05/步 |
| 远离目标 | -0.05/步 |
| 碰撞/越界 | -1.0 |

## 评测指标

- **成功率 (Success Rate)**: 任务成功episode占比
- **SPL (Success weighted by Path Length)**: 成功率加权的路径效率

### 预期目标

| 指标 | Seen场景 | Unseen场景 |
|------|----------|------------|
| 成功率 | > 50% | > 30% |
| SPL | > 0.3 | > 0.2 |

## 相关论文

1. AI2-THOR: *Kolve et al., 2017*
2. RT-2: *Brohan et al., 2023*
3. DD-PPO: *Wijmans et al., 2019*
4. CLIP: *Radford et al., 2021*
5. SAM: *Kirillov et al., 2023*

## 参考资料

- [AI2-THOR官方](https://ai2thor.allenai.org/)
- [Qwen-VL官方](https://github.com/QwenLM/Qwen-VL)
- [Awesome Embodied Robotics](https://github.com/zchoi/Awesome-Embodied-Robotics-and-Agent)

## 课程信息

**课程**: 多模态大模型原理与应用  
**学期**: 2025-2026 春季学期  
**项目类型**: 期末大作业 - 题目5: 游戏或仿真环境中的多模态智能Agent
