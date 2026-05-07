"""VLA骨干模型 - Qwen3-VL 8B + QLoRA"""
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer


class VLABackbone(nn.Module):
    """视觉-语言-动作模型基座

    使用Qwen3-VL 8B作为决策核心，QLoRA 4bit量化微调
    仅从本地路径加载模型，不在线下载
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        model_config = config['model']
        self.base_model_path = model_config.get('base_model', '')

        # 动作token配置
        action_config = model_config['action_tokens']
        self.action_tokens = action_config['tokens']
        self.token_start_idx = action_config['token_start_idx']
        self.num_actions = len(self.action_tokens)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型和tokenizer
        self._load_model(model_config)

    def _load_model(self, model_config: Dict[str, Any]):
        """从本地加载Qwen3-VL模型"""
        quant_config = model_config['quantization']
        lora_config = model_config['lora']

        # 检查本地路径是否存在
        model_path = Path(self.base_model_path)
        if not model_path.exists():
            print(f"Warning: Local model path not found: {self.base_model_path}")
            print("Please run: python scripts/download_model.py --model Qwen/Qwen3-VL-8B-Instruct --output models/Qwen3-VL-8B")
            print("Falling back to dummy model for testing")
            self._init_dummy()
            return

        try:
            # 从本地加载 tokenizer
            print(f"Loading tokenizer from: {self.base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=model_config.get('trust_remote_code', True),
                local_files_only=True,
            )

            # 记录原始词表大小 (用于后续trainable_token_indices计算)
            original_vocab_size = len(self.tokenizer)

            # 添加动作token到词表
            action_token_list = list(self.action_tokens.values())
            num_new_tokens = self.tokenizer.add_tokens(action_token_list)
            print(f"Added {num_new_tokens} action tokens to tokenizer")

            # 从本地加载模型 (使用bitsandbytes量化)
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config['load_in_4bit'],
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
            )

            print(f"Loading model from: {self.base_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                trust_remote_code=model_config.get('trust_remote_code', True),
                local_files_only=True,
                device_map="auto",
            )

            # 调整词表大小 (追加新token的embedding)
            self.model.resize_token_embeddings(len(self.tokenizer))

            # 应用LoRA
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(self.model)

            # 动态计算trainable_token_indices: 新增的action token索引
            trainable_indices = list(range(original_vocab_size, len(self.tokenizer)))
            print(f"Trainable token indices: {trainable_indices}")

            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                lora_dropout=lora_config['lora_dropout'],
                bias=lora_config['bias'],
                task_type=lora_config['task_type'],
                target_modules=lora_config['target_modules'],
                trainable_token_indices=trainable_indices,
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

            self.is_dummy = False

        except Exception as e:
            print(f"Warning: Failed to load local Qwen3-VL model: {e}")
            print("Falling back to dummy model for testing")
            self._init_dummy()

    def _init_dummy(self):
        """初始化dummy模型（用于测试流程）"""
        self.is_dummy = True

        # Dummy tokenizer
        class DummyTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1

            def __call__(self, text, **kwargs):
                ids = torch.randint(0, 100, (1, 10))
                return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

            def decode(self, ids, **kwargs):
                return 'dummy_action'

        self.tokenizer = DummyTokenizer()

        # Dummy model
        self.model = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_actions),
        ).to(self.device)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播

        Args:
            input_ids: 输入token IDs [B, seq_len]
            attention_mask: 注意力掩码 [B, seq_len]
            visual_features: 视觉特征 [B, num_visual_tokens, 4096]

        Returns:
            logits: 动作预测logits [B, num_actions]
        """
        if self.is_dummy:
            # Dummy模型：
            if visual_features is not None:
                # 取平均视觉特征
                vis = visual_features.mean(dim=1)  # [B, 4096]
                logits = self.model(vis)  # [B, num_actions]
            else:
                # 生成需要梯度的 dummy logits (用于测试计算图)
                logits = torch.randn(input_ids.shape[0], self.num_actions, device=self.device, requires_grad=True)
        else:
            # 真实Qwen模型
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # 获取最后一个token的hidden state
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
            last_token_hidden = last_hidden[:, -1, :]  # [B, hidden]

            # 投影到动作空间
            # 使用LM head的转置
            logits = self.model.lm_head(last_token_hidden)  # [B, vocab_size]

            # 只取动作token的logits
            action_token_ids = []
            for token_name in self.action_tokens.values():
                token_id = self.tokenizer.convert_tokens_to_ids(token_name)
                action_token_ids.append(token_id)

            action_token_ids = torch.tensor(action_token_ids, device=self.device)
            logits = logits[:, action_token_ids]  # [B, num_actions]

        return logits

    def generate_action(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       visual_features: Optional[torch.Tensor] = None,
                       temperature: float = 1.0,
                       sample: bool = True) -> int:
        """生成动作

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            visual_features: 视觉特征
            temperature: 采样温度
            sample: 是否采样（False则argmax）

        Returns:
            action_idx: 动作索引 (0-5)
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, visual_features)
            logits = logits / temperature

            if sample:
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.multinomial(probs, num_samples=1).item()
            else:
                action_idx = torch.argmax(logits, dim=-1).item()

        return action_idx

    def get_action_token_ids(self) -> List[int]:
        """获取所有动作token的ID"""
        ids = []
        for token_name in self.action_tokens.values():
            token_id = self.tokenizer.convert_tokens_to_ids(token_name)
            ids.append(token_id)
        return ids

    def save_pretrained(self, save_path: str):
        """保存模型"""
        if self.is_dummy:
            print("Warning: Saving dummy model")
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path / 'model.pt')
        else:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    @classmethod
    def from_pretrained(cls, load_path: str, config: Dict[str, Any]):
        """从保存路径加载模型"""
        # 简化实现
        model = cls(config)
        if not model.is_dummy:
            from peft import PeftModel
            model.model = PeftModel.from_pretrained(model.model, load_path)
        return model
