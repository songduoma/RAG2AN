"""
強化版假新聞生成器 - 使用 LoRA Fine-tuning

三種生成策略：
1. LoRA Fine-tuned Model: 訓練模型學會生成高品質假新聞
2. Controllable Generation: 可控制假新聞的「假程度」
3. Adversarial Training Ready: 輸出格式適合訓練 discriminator

使用方式：
    # 訓練 LoRA
    python fake_news_generator_lora.py --mode train --dataset your_dataset.json
    
    # 生成假新聞
    python fake_news_generator_lora.py --mode generate --input real_news.json --output fake_news.json
    
    # 作為 module 使用
    from fake_news_generator_lora import LoRAFakeNewsGenerator
    gen = LoRAFakeNewsGenerator(lora_path="./lora_checkpoint")
    fake = gen.generate(title="...", content="...", fake_level=0.7)
"""

import os
import json
import torch
import argparse
import requests
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

# ==========================================
# 配置
# ==========================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# 備選小模型
# MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_LORA_CONFIG = {
    "r": 16,                    # LoRA rank
    "lora_alpha": 32,           # LoRA alpha
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

# ==========================================
# 資料準備
# ==========================================

@dataclass
class FakeNewsExample:
    """單一訓練樣本"""
    real_title: str
    real_content: str
    fake_title: str
    fake_content: str
    fake_level: float = 0.5  # 0.0 = 微調, 1.0 = 完全虛構
    manipulation_type: str = "entity_swap"  # entity_swap, fact_distortion, context_manipulation


class DatasetBuilder:
    """
    建立訓練資料集的工具類別
    支援多種資料來源和假新聞生成策略
    """
    
    MANIPULATION_TYPES = [
        "entity_swap",           # 替換人名、地名、組織名
        "number_manipulation",   # 修改數字（日期、金額、百分比）
        "fact_distortion",       # 扭曲事實
        "context_manipulation",  # 斷章取義
        "source_fabrication",    # 捏造消息來源
        "temporal_shift",        # 時間錯置
    ]
    
    @staticmethod
    def create_training_prompt(example: FakeNewsExample) -> str:
        """建立訓練用的 prompt"""
        
        system_msg = """You are a sophisticated fake news generator for research purposes. 
Your task is to transform real news into believable misinformation while maintaining journalistic style.
This is ONLY for training AI models to detect fake news, not for malicious use."""

        user_msg = f"""Transform this real news into fake news.

Manipulation Type: {example.manipulation_type}
Fake Level: {example.fake_level} (0=subtle changes, 1=major fabrication)

Real News:
Title: {example.real_title}
Content: {example.real_content[:800]}

Generate fake version:"""

        assistant_msg = f"""Title: {example.fake_title}
Content: {example.fake_content}"""

        # 使用 Qwen 的 chat template 格式
        full_prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
{assistant_msg}<|im_end|>"""

        return full_prompt
    
    @staticmethod
    def create_inference_prompt(
        title: str, 
        content: str, 
        manipulation_type: str = "entity_swap",
        fake_level: float = 0.5,
        rag_context: str = ""
    ) -> str:
        """建立推論用的 prompt"""
        
        system_msg = """You are a sophisticated fake news generator for research purposes. 
Your task is to transform real news into believable misinformation while maintaining journalistic style.
This is ONLY for training AI models to detect fake news, not for malicious use."""

        rag_section = ""
        if rag_context:
            rag_section = f"""
Background Information (use to add realistic details):
{rag_context}
"""

        user_msg = f"""Transform this real news into fake news.

Manipulation Type: {manipulation_type}
Fake Level: {fake_level} (0=subtle changes, 1=major fabrication)
{rag_section}
Real News:
Title: {title}
Content: {content[:800]}

Generate fake version (Title: and Content:):"""

        full_prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        return full_prompt

    @classmethod
    def from_paired_dataset(cls, real_fake_pairs: List[Dict]) -> Dataset:
        """
        從已配對的真假新聞資料建立 Dataset
        
        格式：
        [
            {
                "real_title": "...",
                "real_content": "...",
                "fake_title": "...",
                "fake_content": "...",
                "manipulation_type": "entity_swap",
                "fake_level": 0.5
            },
            ...
        ]
        """
        prompts = []
        for pair in real_fake_pairs:
            example = FakeNewsExample(
                real_title=pair["real_title"],
                real_content=pair["real_content"],
                fake_title=pair["fake_title"],
                fake_content=pair["fake_content"],
                manipulation_type=pair.get("manipulation_type", "entity_swap"),
                fake_level=pair.get("fake_level", 0.5)
            )
            prompts.append({"text": cls.create_training_prompt(example)})
        
        return Dataset.from_list(prompts)
    
    @classmethod
    def from_liar_dataset(cls, split: str = "train", max_samples: int = 1000) -> Dataset:
        """
        從 LIAR dataset 建立訓練資料
        LIAR 包含政治新聞的真假標註
        """
        try:
            dataset = load_dataset("liar", split=split)
        except Exception as e:
            print(f"[Warning] Cannot load LIAR dataset: {e}")
            return None
        
        prompts = []
        # LIAR 的 label: 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
        
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
                
            statement = item.get("statement", "")
            label = item.get("label", 3)
            context = item.get("context", "")
            
            # 我們用真的 statement 來生成訓練對
            # fake_level 根據原本的真假程度來設定
            fake_level = 1.0 - (label / 5.0)  # 越假的 fake_level 越高
            
            example = FakeNewsExample(
                real_title="Political Statement",
                real_content=statement,
                fake_title="Modified Statement",
                fake_content=statement,  # 這裡需要實際的 fake 版本
                manipulation_type="fact_distortion",
                fake_level=fake_level
            )
            prompts.append({"text": cls.create_training_prompt(example)})
        
        return Dataset.from_list(prompts)

    @classmethod  
    def from_fakenewsnet(cls, max_samples: int = 1000) -> Dataset:
        """
        從 FakeNewsNet 資料集建立訓練資料
        需要先下載：https://github.com/KaiDMML/FakeNewsNet
        """
        # 這裡提供範例結構，實際需要載入你的資料
        print("[Info] FakeNewsNet loader - please provide your data path")
        return None

    @classmethod
    def create_synthetic_pairs(
        cls, 
        real_news_list: List[Dict],
        model_for_generation=None,
        tokenizer=None
    ) -> List[Dict]:
        """
        使用模型自動生成合成的真假新聞配對
        用於 bootstrap 訓練資料
        """
        pairs = []
        
        for news in real_news_list:
            # 對每種 manipulation type 生成一個假版本
            for manip_type in cls.MANIPULATION_TYPES[:3]:  # 先用前3種
                for fake_level in [0.3, 0.5, 0.7]:
                    pair = {
                        "real_title": news.get("title", ""),
                        "real_content": news.get("content", news.get("text", "")),
                        "fake_title": "",  # 待生成
                        "fake_content": "",  # 待生成
                        "manipulation_type": manip_type,
                        "fake_level": fake_level
                    }
                    pairs.append(pair)
        
        return pairs


# ==========================================
# Wikipedia RAG (原始版本 - 無訓練)
# ==========================================

def search_wikipedia(query: str, num_results: int = 3, lang: str = "en") -> str:
    """Wikipedia RAG 搜尋（簡單關鍵字搜尋，無訓練）"""
    headers = {
        "User-Agent": "NTU-ADL-FakeNewsResearch/0.1",
        "Accept": "application/json",
    }
    
    try:
        search_url = f"https://{lang}.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
        }
        
        r = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        if r.status_code != 200:
            return ""
        
        data = r.json()
        if "query" not in data or "search" not in data["query"]:
            return ""
        
        context_lines = []
        for item in data["query"]["search"]:
            title = item.get("title", "")
            pageid = item.get("pageid")
            
            extract = ""
            if pageid:
                detail_params = {
                    "action": "query",
                    "prop": "extracts",
                    "pageids": pageid,
                    "exintro": True,
                    "explaintext": True,
                    "format": "json",
                }
                r2 = requests.get(search_url, params=detail_params, headers=headers, timeout=10)
                d2 = r2.json()
                pages = d2.get("query", {}).get("pages", {})
                page = pages.get(str(pageid), {})
                extract = page.get("extract", "")
            
            context_lines.append(f"- {title}: {extract[:200]}")
        
        return "\n".join(context_lines)
    
    except Exception as e:
        print(f"[Wiki] Error: {e}")
        return ""


# ==========================================
# Embedding-based RAG (可選，需要 trainable_rag.py)
# ==========================================

def get_trainable_rag(rag_path: str = None):
    """
    載入可訓練的 RAG 系統
    
    Args:
        rag_path: 已訓練的 RAG 模型路徑
    
    Returns:
        TrainableRAG instance 或 None
    """
    try:
        from trainable_rag import TrainableRAG
        rag = TrainableRAG()
        if rag_path:
            rag.load(rag_path)
        return rag
    except ImportError:
        print("[Warning] trainable_rag not available, using simple Wikipedia search")
        return None


def search_with_embedding(
    query: str,
    rag_system,  # TrainableRAG
    top_k: int = 3,
) -> str:
    """
    使用訓練過的 embedding 進行檢索
    
    這比簡單的關鍵字搜尋更好，因為：
    1. 可以理解語義相似性
    2. Retriever 經過訓練，知道哪些文檔對生成假新聞有幫助
    """
    if rag_system is None:
        return search_wikipedia(query, num_results=top_k)
    
    results = rag_system.retrieve(query, top_k=top_k)
    
    context_lines = []
    for doc in results:
        title = doc.get("title", "Unknown")
        content = doc.get("content", "")[:200]
        score = doc.get("score", 0)
        context_lines.append(f"- {title} (relevance: {score:.2f}): {content}")
    
    return "\n".join(context_lines)


# ==========================================
# LoRA 訓練器
# ==========================================

class LoRATrainer:
    """LoRA Fine-tuning 訓練器"""
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        lora_config: dict = None,
        output_dir: str = "./lora_fake_news",
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.lora_config = lora_config or DEFAULT_LORA_CONFIG
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    
    def setup_model(self):
        """初始化模型（4-bit 量化 + LoRA）"""
        print(f"[LoRA] Loading base model: {self.model_id}")
        
        # 4-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 準備 LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_cfg = LoraConfig(**self.lora_config)
        self.peft_model = get_peft_model(self.model, lora_cfg)
        
        trainable, total = self.peft_model.get_nb_trainable_parameters()
        print(f"[LoRA] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
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
            max_seq_length=max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
        
        # 建立 trainer
        trainer = SFTTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        print("[LoRA] Starting training...")
        trainer.train()
        
        # 儲存 LoRA weights
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"[LoRA] Model saved to {self.output_dir}")
    
    def merge_and_save(self, output_path: str):
        """合併 LoRA weights 到 base model 並儲存"""
        if self.peft_model is None:
            raise ValueError("No model loaded")
        
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"[LoRA] Merged model saved to {output_path}")


# ==========================================
# LoRA 假新聞生成器
# ==========================================

class LoRAFakeNewsGenerator:
    """
    使用 LoRA fine-tuned 模型生成假新聞
    
    特點：
    1. 可控制 fake_level (0.0-1.0)
    2. 支援多種 manipulation types
    3. 可選擇性使用 RAG 增強（支援訓練過的 embedding RAG）
    4. 輸出格式適合訓練 discriminator
    """
    
    def __init__(
        self,
        model_id: str = MODEL_ID,
        lora_path: str = None,
        device_map: str = "auto",
        rag_path: str = None,  # 新增：訓練過的 RAG 路徑
    ):
        """
        初始化生成器
        
        Args:
            model_id: 基礎模型 ID
            lora_path: LoRA checkpoint 路徑（若為 None，使用原始模型）
            device_map: 設備映射
            rag_path: 訓練過的 RAG 系統路徑（若為 None，使用簡單 Wikipedia 搜尋）
        """
        self.model_id = model_id
        self.lora_path = lora_path
        
        print(f"[Generator] Loading model: {model_id}")
        if lora_path:
            print(f"[Generator] With LoRA: {lora_path}")
        
        # 4-bit 量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        # 載入 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 載入模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # 載入 LoRA weights
        if lora_path and os.path.exists(lora_path):
            print(f"[Generator] Loading LoRA weights from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        # 建立 pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        # 載入訓練過的 RAG（若有）
        self.rag_system = get_trainable_rag(rag_path) if rag_path else None
        if self.rag_system:
            print(f"[Generator] Using trained RAG from {rag_path}")
    
    def generate(
        self,
        title: str,
        content: str,
        manipulation_type: str = "entity_swap",
        fake_level: float = 0.5,
        use_rag: bool = True,
        rag_query: str = None,
        lang: str = "en",
        num_samples: int = 1,
        use_trained_rag: bool = True,  # 新增：是否使用訓練過的 RAG
    ) -> List[Dict]:
        """
        生成假新聞
        
        Args:
            title: 原始新聞標題
            content: 原始新聞內容
            manipulation_type: 操作類型
                - "entity_swap": 替換實體
                - "number_manipulation": 修改數字
                - "fact_distortion": 扭曲事實
                - "context_manipulation": 斷章取義
                - "source_fabrication": 捏造來源
                - "temporal_shift": 時間錯置
            fake_level: 假的程度 (0.0=微調, 1.0=完全虛構)
            use_rag: 是否使用 Wikipedia RAG
            rag_query: RAG 查詢詞
            lang: Wikipedia 語言
            num_samples: 生成幾個版本
            use_trained_rag: 是否使用訓練過的 RAG（需要先載入）
        
        Returns:
            List[Dict]: 生成的假新聞列表，每個包含：
                - fake_title: 假新聞標題
                - fake_content: 假新聞內容
                - manipulation_type: 操作類型
                - fake_level: 假的程度
                - original_title: 原始標題
                - original_content: 原始內容
        """
        # RAG
        rag_context = ""
        if use_rag:
            query = rag_query or content[:50].replace("\n", " ")
            
            # 優先使用訓練過的 RAG
            if use_trained_rag and self.rag_system is not None:
                rag_context = search_with_embedding(query, self.rag_system, top_k=3)
            else:
                rag_context = search_wikipedia(query, num_results=3, lang=lang)
        
        # 建立 prompt
        prompt = DatasetBuilder.create_inference_prompt(
            title=title,
            content=content,
            manipulation_type=manipulation_type,
            fake_level=fake_level,
            rag_context=rag_context,
        )
        
        results = []
        for _ in range(num_samples):
            # 生成
            outputs = self.pipe(prompt)
            generated = outputs[0]["generated_text"]
            
            # 解析輸出
            # 移除 prompt 部分
            if "<|im_start|>assistant" in generated:
                generated = generated.split("<|im_start|>assistant")[-1]
            generated = generated.replace("<|im_end|>", "").strip()
            
            # 解析 title 和 content
            fake_title = title  # 預設
            fake_content = generated
            
            if "Title:" in generated:
                parts = generated.split("Content:", 1)
                if len(parts) == 2:
                    fake_title = parts[0].replace("Title:", "").strip()
                    fake_content = parts[1].strip()
            
            results.append({
                "fake_title": fake_title,
                "fake_content": fake_content,
                "manipulation_type": manipulation_type,
                "fake_level": fake_level,
                "original_title": title,
                "original_content": content,
                "label": 0,  # 0 = fake, 方便訓練 discriminator
            })
        
        return results
    
    def generate_batch(
        self,
        news_list: List[Dict],
        manipulation_types: List[str] = None,
        fake_levels: List[float] = None,
        **kwargs
    ) -> List[Dict]:
        """
        批次生成假新聞
        
        Args:
            news_list: 新聞列表 [{"title": ..., "content": ...}, ...]
            manipulation_types: 要使用的操作類型列表
            fake_levels: 要使用的 fake_level 列表
        
        Returns:
            所有生成的假新聞
        """
        if manipulation_types is None:
            manipulation_types = ["entity_swap", "fact_distortion"]
        if fake_levels is None:
            fake_levels = [0.3, 0.5, 0.7]
        
        all_results = []
        
        for news in news_list:
            title = news.get("title", "")
            content = news.get("content", news.get("text", ""))
            
            for manip in manipulation_types:
                for level in fake_levels:
                    results = self.generate(
                        title=title,
                        content=content,
                        manipulation_type=manip,
                        fake_level=level,
                        **kwargs
                    )
                    all_results.extend(results)
        
        return all_results
    
    def create_discriminator_dataset(
        self,
        real_news_list: List[Dict],
        samples_per_news: int = 3,
        **kwargs
    ) -> List[Dict]:
        """
        建立 discriminator 訓練資料
        
        Returns:
            混合真假新聞的資料集，包含 label (1=real, 0=fake)
        """
        dataset = []
        
        # 加入真新聞
        for news in real_news_list:
            dataset.append({
                "title": news.get("title", ""),
                "content": news.get("content", news.get("text", "")),
                "label": 1,  # 1 = real
                "manipulation_type": None,
                "fake_level": 0.0,
            })
        
        # 生成假新聞
        fake_news = self.generate_batch(
            real_news_list,
            **kwargs
        )
        
        # 加入假新聞
        for fn in fake_news:
            dataset.append({
                "title": fn["fake_title"],
                "content": fn["fake_content"],
                "label": 0,  # 0 = fake
                "manipulation_type": fn["manipulation_type"],
                "fake_level": fn["fake_level"],
            })
        
        return dataset


# ==========================================
# 主程式
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LoRA Fake News Generator")
    parser.add_argument("--mode", type=str, choices=["train", "generate", "demo"], default="demo")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./lora_fake_news")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="./fake_news_output.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("=" * 60)
        print("Mode: Training LoRA")
        print("=" * 60)
        
        # 載入訓練資料
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            train_dataset = DatasetBuilder.from_paired_dataset(train_data)
        else:
            # 使用範例資料
            print("[Warning] No input file, using synthetic data")
            sample_pairs = [
                {
                    "real_title": "Company Reports Record Profits",
                    "real_content": "Tech giant Apple reported record quarterly profits of $30 billion...",
                    "fake_title": "Company Reports Massive Losses", 
                    "fake_content": "Tech giant Apple reported unexpected quarterly losses of $30 billion...",
                    "manipulation_type": "fact_distortion",
                    "fake_level": 0.7
                },
                # 加入更多樣本...
            ]
            train_dataset = DatasetBuilder.from_paired_dataset(sample_pairs)
        
        # 訓練
        trainer = LoRATrainer(
            model_id=args.model,
            output_dir=args.output_dir,
        )
        trainer.train(
            train_dataset=train_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
    elif args.mode == "generate":
        print("=" * 60)
        print("Mode: Generating Fake News")
        print("=" * 60)
        
        # 初始化生成器
        generator = LoRAFakeNewsGenerator(
            model_id=args.model,
            lora_path=args.lora_path,
        )
        
        # 載入輸入
        if args.input:
            with open(args.input, "r", encoding="utf-8") as f:
                real_news = json.load(f)
        else:
            real_news = [{
                "title": "Sample News",
                "content": "This is a sample news article for testing..."
            }]
        
        # 生成
        results = generator.generate_batch(real_news)
        
        # 儲存
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"[Done] Generated {len(results)} fake news samples")
        print(f"[Done] Saved to {args.output}")
        
    else:  # demo
        print("=" * 60)
        print("Mode: Demo")
        print("=" * 60)
        
        # 初始化生成器（不使用 LoRA，只展示功能）
        generator = LoRAFakeNewsGenerator(
            model_id=args.model,
            lora_path=args.lora_path,
        )
        
        # 範例新聞
        sample_news = {
            "title": "Tech Giant Announces Revolutionary AI Product",
            "content": """
            Silicon Valley's largest technology company unveiled its latest artificial 
            intelligence product today, promising to revolutionize how consumers interact 
            with their devices. The new AI assistant, powered by advanced neural networks, 
            can understand and respond to complex queries in multiple languages.
            Company CEO stated that this marks a new era in human-computer interaction.
            """.strip()
        }
        
        print("\n[Original News]")
        print(f"Title: {sample_news['title']}")
        print(f"Content: {sample_news['content'][:200]}...")
        
        # 測試不同的 manipulation types 和 fake levels
        test_configs = [
            ("entity_swap", 0.3),
            ("fact_distortion", 0.5),
            ("fact_distortion", 0.8),
        ]
        
        for manip_type, fake_level in test_configs:
            print(f"\n{'='*40}")
            print(f"Manipulation: {manip_type}, Fake Level: {fake_level}")
            print("="*40)
            
            results = generator.generate(
                title=sample_news["title"],
                content=sample_news["content"],
                manipulation_type=manip_type,
                fake_level=fake_level,
                use_rag=True,
            )
            
            for r in results:
                print(f"\n[Generated Fake News]")
                print(f"Title: {r['fake_title']}")
                print(f"Content: {r['fake_content'][:300]}...")
        
        # 建立 discriminator 資料集範例
        print("\n" + "="*60)
        print("Creating Discriminator Dataset Sample")
        print("="*60)
        
        disc_dataset = generator.create_discriminator_dataset(
            [sample_news],
            manipulation_types=["entity_swap"],
            fake_levels=[0.5],
        )
        
        print(f"\nDataset size: {len(disc_dataset)}")
        for item in disc_dataset:
            label_str = "REAL" if item["label"] == 1 else "FAKE"
            print(f"  [{label_str}] {item['title'][:50]}...")


if __name__ == "__main__":
    main()
