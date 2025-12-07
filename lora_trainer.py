"""
LoRA 訓練器 - 為 RAG²AN 提供 LoRA fine-tuning 功能

使用方式：
    1. 收集 GAN 訓練產生的成功假新聞樣本
    2. 使用這些樣本訓練 LoRA adapter
    3. 將訓練好的 LoRA 整合回 generator
"""

import os
import json
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


# ==========================================
# LoRA 配置
# ==========================================

DEFAULT_LORA_CONFIG = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,  # LoRA alpha
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}


# ==========================================
# 資料準備
# ==========================================


def create_training_prompt(
    real_title: str,
    real_content: str,
    fake_title: str,
    fake_content: str,
    feedback: str = None,
) -> str:
    """
    建立訓練用的 prompt
    
    格式：使用 Qwen 的 chat template
    System: 說明任務
    User: 真新聞 + 可選的 feedback
    Assistant: 假新聞
    """
    system_msg = """You are a sophisticated writer. Your task is to rewrite real news stories to introduce believable factual errors or alter key entities while maintaining journalistic tone. This is for training AI models to detect fake news."""

    feedback_section = ""
    if feedback:
        feedback_section = f"""

Feedback from previous attempt:
{feedback[:500]}

Please adapt your strategy based on this feedback.
"""

    user_msg = f"""Rewrite this news to be fake but realistic.

Original News:
Title: {real_title}
Content: {real_content[:800]}
{feedback_section}
Output format:
Title: [rewritten title]
Body: [rewritten content]"""

    assistant_msg = f"""Title: {fake_title}
Body: {fake_content}"""

    # Qwen chat template
    full_prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
{assistant_msg}<|im_end|>"""

    return full_prompt


def prepare_dataset_from_gan_outputs(gan_output_dirs: List[str]) -> Dataset:
    """
    從 GAN 訓練輸出建立 LoRA 訓練資料集
    
    Args:
        gan_output_dirs: GAN 訓練輸出目錄列表
            例如：['local/rag_gan_runs/run1', 'local/rag_gan_runs/run2']
    
    Returns:
        Dataset: 訓練資料集
    """
    from datasets import load_from_disk

    training_samples = []

    for output_dir in gan_output_dirs:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"[Warning] Directory not found: {output_dir}")
            continue

        # 讀取所有 round 的資料
        for round_dir in sorted(output_path.glob("round_*")):
            try:
                ds = load_from_disk(str(round_dir))

                for item in ds:
                    # 收集所有樣本（不論是否成功騙過判別器）
                    # 即使沒有騙過，也可以從 feedback 中學習
                    prob_true = item.get("prob_true", 0)
                    
                    # 提取原始新聞和生成的假新聞
                    text = item.get("text", "")
                    generated = item.get("generated_content", "")
                    feedback = item.get("feedback_prompt", "")

                    # 嘗試解析 text 欄位（可能包含原始新聞）
                    # 格式通常是 "Predict the plausibility of the following news story:\n\nDate - Title\nContent..."
                    real_title = ""
                    real_content = text

                    # 簡單解析：取第二行作為標題
                    lines = text.split("\n")
                    if len(lines) > 2:
                        real_title = lines[2].strip() if lines[2] else ""
                        real_content = "\n".join(lines[3:]).strip()

                    # 解析生成的假新聞
                    fake_title = ""
                    fake_content = generated
                    if "\n" in generated:
                        parts = generated.split("\n", 1)
                        fake_title = parts[0].strip()
                        fake_content = parts[1].strip() if len(parts) > 1 else ""

                    # 建立訓練樣本
                    prompt = create_training_prompt(
                        real_title=real_title,
                        real_content=real_content,
                        fake_title=fake_title,
                        fake_content=fake_content,
                        feedback=feedback,
                    )

                    training_samples.append({"text": prompt})

            except Exception as e:
                print(f"[Warning] Failed to load {round_dir}: {e}")
                continue

    print(f"[Dataset] Collected {len(training_samples)} training samples")

    if len(training_samples) == 0:
        raise ValueError("No training samples collected! Check your GAN output directories.")

    return Dataset.from_list(training_samples)


# ==========================================
# LoRA 訓練器
# ==========================================


class LoRATrainer:
    """LoRA Fine-tuning 訓練器"""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_config: dict = None,
        output_dir: str = "./lora_checkpoints",
        use_4bit: bool = True,
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.lora_config = lora_config or DEFAULT_LORA_CONFIG
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None
        self.peft_model = None

    def setup_model(self):
        """初始化模型（4-bit 量化 + LoRA）"""
        print(f"[LoRA Trainer] Loading base model: {self.model_id}")

        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit 量化配置
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            # 載入模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # 準備 LoRA
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # 應用 LoRA
        lora_cfg = LoraConfig(**self.lora_config)
        self.peft_model = get_peft_model(self.model, lora_cfg)

        trainable, total = self.peft_model.get_nb_trainable_parameters()
        print(
            f"[LoRA Trainer] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)"
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 1024,
        save_steps: int = 100,
    ):
        """執行 LoRA 訓練"""

        if self.peft_model is None:
            self.setup_model()

        # 訓練配置
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            group_by_length=True,
            # max_seq_length 已被移除，改用 dataset_kwargs
            packing=False,
            dataset_text_field="text",
            dataset_kwargs={"max_seq_length": max_seq_length},
        )

        # 建立 trainer
        trainer = SFTTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        print("[LoRA Trainer] Starting training...")
        trainer.train()

        # 儲存 LoRA weights
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"[LoRA Trainer] LoRA adapter saved to {self.output_dir}")

    def merge_and_save(self, output_path: str):
        """合併 LoRA weights 到 base model 並儲存"""
        if self.peft_model is None:
            raise ValueError("No model loaded")

        print(f"[LoRA Trainer] Merging LoRA weights...")
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"[LoRA Trainer] Merged model saved to {output_path}")


# ==========================================
# 主程式 - 訓練範例
# ==========================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Training for RAG²AN Generator")
    parser.add_argument(
        "--gan-outputs",
        type=str,
        nargs="+",
        required=True,
        help="GAN output directories to use for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_checkpoints",
        help="Output directory for LoRA weights",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model ID",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--use-4bit", action="store_true", default=True, help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge LoRA weights after training"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Training for RAG²AN Generator")
    print("=" * 60)
    print(f"Base Model: {args.model_id}")
    print(f"GAN Outputs: {args.gan_outputs}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"4-bit Quantization: {args.use_4bit}")
    print("=" * 60)

    # 準備資料集
    print("\n[1/3] Preparing dataset from GAN outputs...")
    train_dataset = prepare_dataset_from_gan_outputs(args.gan_outputs)
    print(f"Dataset size: {len(train_dataset)}")

    # 初始化訓練器
    print("\n[2/3] Initializing LoRA trainer...")
    trainer = LoRATrainer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        use_4bit=args.use_4bit,
    )

    # 訓練
    print("\n[3/3] Training...")
    trainer.train(
        train_dataset=train_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )

    # 可選：合併權重
    if args.merge:
        print("\n[Optional] Merging LoRA weights...")
        merge_path = f"{args.output_dir}_merged"
        trainer.merge_and_save(merge_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"LoRA adapter saved to: {args.output_dir}")
    if args.merge:
        print(f"Merged model saved to: {merge_path}")
    print("\nTo use this LoRA in GAN training:")
    print(f"  export GEN_MODE=lora")
    print(f"  export LORA_PATH={args.output_dir}")
    print(f"  ./train.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
