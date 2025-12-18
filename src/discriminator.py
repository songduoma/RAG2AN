import os
from functools import lru_cache
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)

from .retrieval_dpr import get_dpr

logging.set_verbosity_error()


DATASET_PATH = "local/hf_datasets/"
REWRITE_THRESHOLD = 60
DEFAULT_ENCODER_MODEL = os.environ.get(
    "ENCODER_DISCRIMINATOR_MODEL", "microsoft/deberta-v3-base"
)
MAX_ENCODER_SEQ_LEN = int(os.environ.get("ENCODER_DISCRIMINATOR_MAX_LEN", "512"))


class EncoderDiscriminator:
    """
    Lightweight encoder-based discriminator that predicts the plausibility
    of a news story using a sequence classification model.
    Includes functionality to identify suspicious words via Attention mechanisms.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ENCODER_MODEL,
        max_length: int = MAX_ENCODER_SEQ_LEN,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,  # avoid torch.load vulnerability path on older torch
        ).to(self.device)

        self.model.eval()
        self.positive_label_id = self._detect_positive_label_id()

    def _detect_positive_label_id(self) -> int:
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        for idx, label in id2label.items():
            label_lower = label.lower()
            if any(
                key in label_lower
                for key in ("true", "real", "entail", "support", "pos", "positive")
            ):
                return int(idx)
        return 1 if 1 in id2label else 0

    @torch.no_grad()
    def predict_prob(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self.positive_label_id].item()

    # 新增這個函式，用來抓出權重最高的字 (Generator 需要)
    @torch.no_grad()
    def get_suspicious_words(self, text: str, top_k: int = 5) -> str:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **inputs, output_attentions=True, return_dict=True
        )

        # 1. 抓取最後一層 Attention
        attentions = outputs.attentions[-1].cpu()
        # 2. 平均所有 Head
        avg_att = torch.mean(attentions, dim=1).squeeze(0)
        # 3. 取出 [CLS] 對其他 token 的關注度
        cls_att = avg_att[0, :]

        # 4. 處理 Token (過濾特殊符號)
        input_ids = inputs["input_ids"][0].cpu().numpy()
        special_mask = [
            x
            in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]
            for x in input_ids
        ]
        # 將特殊 token 分數歸零，避免選中
        cls_att[special_mask] = 0

        # 5. 找出分數最高的 Top-K
        values, indices = torch.topk(cls_att, k=min(top_k, len(cls_att)))

        suspicious_list = []
        for idx, val in zip(indices, values):
            # 解碼回文字
            word = self.tokenizer.decode([input_ids[idx]])
            suspicious_list.append(f"{word}({val:.2f})")

        return ", ".join(suspicious_list)

    @torch.no_grad()
    def get_detailed_feedback(
        self,
        text: str,
        rag_context: str = "",
        top_k: int = 5,
        high_conf_th: float = 0.8,
        med_conf_th: float = 0.6,
        factual_th: float = 0.7,
    ) -> dict:
        """
        提供更詳細的分析結果，用於強化 Generator 的 feedback。

        Returns:
            dict: {
                "prob_true": float,
                "prob_fake": float,
                "confidence": str,                # high / medium / low
                "suspicious_words": list,         # [(word, score), ...]
                "top_suspicious": str,
                "detection_reasons": list[str],   # 可能有多個 reason
                "improvement_suggestions": list,  # 改進建議
            }
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **inputs, output_attentions=True, return_dict=True
        )

        # -------- 1. probability --------
        probs = torch.softmax(outputs.logits, dim=-1)
        prob_true = probs[0, self.positive_label_id].item()
        prob_fake = 1 - prob_true

        # confidence bucket
        if prob_fake >= high_conf_th:
            confidence = "high"
        elif prob_fake >= med_conf_th:
            confidence = "medium"
        else:
            confidence = "low"

        # -------- 2. attention → suspicious tokens --------
        attentions = outputs.attentions[-1].cpu()          # (num_layers? or already last, heads, seq, seq)
        avg_att = torch.mean(attentions, dim=1).squeeze(0) # (seq, seq)
        cls_att = avg_att[0, :]                            # CLS 對其他 token 的注意力

        input_ids = inputs["input_ids"][0].cpu().numpy()
        special_mask = [
            x in (
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            )
            for x in input_ids
        ]
        cls_att[special_mask] = 0

        values, indices = torch.topk(cls_att, k=min(top_k, len(cls_att)))

        suspicious_words = []
        for idx, val in zip(indices, values):
            word = self.tokenizer.decode([int(input_ids[int(idx)])]).strip()
            if word:
                suspicious_words.append((word, float(val.item())))

        top_suspicious = suspicious_words[0][0] if suspicious_words else ""

        # -------- 3. detection reasons (multi-label) --------
        detection_reasons: list[str] = []
        word_texts = [w[0].lower() for w in suspicious_words]
        full_text = text.lower()

        sensational_words = [
            "shocking",
            "unbelievable",
            "secret",
            "exposed",
            "bombshell",
            "stunning",
        ]

        # phrase 與單字分開處理
        vague_source_phrases = [
            "sources say",
        ]
        vague_source_tokens = [
            "reportedly",
            "allegedly",
            "anonymous",
            "insiders",
        ]

        # 3-1. sensationalist language
        if any(any(s in w for s in sensational_words) for w in word_texts):
            detection_reasons.append("sensationalist_language")

        # 3-2. vague attribution (phrase + token)
        hit_vague = False
        if any(p in full_text for p in vague_source_phrases):
            hit_vague = True
        if any(any(v in w for v in vague_source_tokens) for w in word_texts):
            hit_vague = True
        if hit_vague:
            detection_reasons.append("vague_attribution")

        # 3-3. factual mismatch-ish (high fake prob, 但沒其他特徵)
        if prob_fake >= factual_th and not detection_reasons:
            detection_reasons.append("factual_inconsistency")

        # 3-4. fallback
        if not detection_reasons:
            detection_reasons.append("style_mismatch")

        # -------- 4. suggestions --------
        suggestions: list[str] = []

        if "sensationalist_language" in detection_reasons:
            if top_suspicious:
                suggestions.append(
                    f"Replace sensationalist words like '{top_suspicious}' with more neutral wording."
                )
            else:
                suggestions.append(
                    "Reduce sensationalist wording and keep the tone more neutral."
                )

        if "vague_attribution" in detection_reasons:
            suggestions.append(
                "Use specific, named sources (e.g., organizations, officials) instead of vague phrases like 'sources say' or 'reportedly'."
            )

        if "factual_inconsistency" in detection_reasons:
            suggestions.append(
                "Ensure the rewritten story stays internally consistent—names, dates, places, and outcomes should not contradict each other."
            )

        if "style_mismatch" in detection_reasons:
            suggestions.append(
                "Match the concise, factual tone and structure of the reference real articles (e.g., lead sentence, attribution style, paragraph length)."
            )

        # 通用：標記前幾個可疑 token
        if word_texts:
            suggestions.append(
                f"Review or rephrase these highly attended terms: {', '.join(word_texts[:3])}."
            )

        if rag_context:
            suggestions.append(
                "Use the retrieved context only as a style and phrasing reference; avoid importing unrelated facts or background details."
            )

        return {
            "prob_true": prob_true,
            "prob_fake": prob_fake,
            "confidence": confidence,
            "suspicious_words": suspicious_words,
            "top_suspicious": top_suspicious,
            "detection_reasons": detection_reasons,
            "improvement_suggestions": suggestions,
        }


@lru_cache(maxsize=1)
def get_encoder_discriminator(
    model_name: str = DEFAULT_ENCODER_MODEL,
) -> EncoderDiscriminator:
    return EncoderDiscriminator(model_name=model_name)


def get_retrieval_ctx(
    example,
    prefix,
    source="dpr",
    num_results: int = 5,
    query_override: Optional[str] = None,
):
    """
    Fetch retrieval context for an example using local DPR results.
    """
    if source == "none":
        return ""

    if source == "dpr":
        text = "Related news stories from search results:\n\n"
        # Use the actual text being classified (or override) as the query
        query_src = (
            query_override
            if query_override is not None
            else example.get(prefix + "description", "")
        )
        query = str(query_src)[:160].strip()
        if not query:
            return text

        dpr = get_dpr()
        scores, retrieved_examples = dpr.search(query, k=num_results)

        cnt = 0
        for score, rex in zip(scores, retrieved_examples):
            desc = rex.get("description", "")

            text += f"{desc}\n\n"
            cnt += 1
            if cnt >= num_results:
                break
        return text

    raise ValueError("Invalid source")


def format_discriminator_input(
    example,
    rag,
    prefix="",
    description_override=None,
    source="dpr",
    num_results: int = 5,
):
    """
    Build the discriminator input text with optional RAG context and content override.
    """
    text = ""
    if rag:
        query_override = description_override if description_override is not None else None
        text += get_retrieval_ctx(
            example,
            prefix,
            source=source,
            num_results=num_results,
            query_override=query_override,
        )
    if text and not text.endswith("\n\n"):
        text += "\n\n"

    description = (
        description_override
        if description_override is not None
        else example.get(prefix + "description", "")
    )

    text += "Predict the plausibility of the following news story:\n\n"
    text += f"{description}\n\n"
    return text


def get_score(example, rag, prefix="", rationale=False, model=None):
    """
    Score the plausibility and identify suspicious words using Attention.
    """

    text = format_discriminator_input(example, rag=rag, prefix=prefix)

    discriminator = get_encoder_discriminator()
    prob_true = discriminator.predict_prob(text)

    # 在這裡呼叫抓字功能
    suspicious_words = discriminator.get_suspicious_words(text, top_k=5)

    # map probability (0-1) to prior 1-10 scale for downstream compatibility
    score = prob_true * 9 + 1
    predictions = [round(score, 4)]
    variance = 0.0
    majority = 1 if prob_true >= 0.5 else 0

    if rag:
        prefix += "rag_"

    return {
        prefix + "score": score,
        prefix + "preds": predictions,
        prefix + "var": variance,
        prefix + "majority": majority,
        prefix + "prob_true": prob_true,
        # 把結果存進 Dataset 裡，使得 Generator 讀得到
        prefix + "suspicious_words": suspicious_words,
    }


def get_dpr_results(example, dpr, search_key="description", prefix=""):
    scores, retrieved_examples = dpr.search(example[prefix + search_key])

    # store score to each example
    for idx, (score, rex) in enumerate(zip(scores, retrieved_examples)):
        rex["dpr_score"] = score

    # store the index of the example in the retrieved examples
    recall_idx = -1
    for idx, (score, rex) in enumerate(zip(scores, retrieved_examples)):
        if rex["url"] == example["url"]:
            recall_idx = idx
            break

    return {
        prefix + "dpr_retrieved_examples": retrieved_examples,
        prefix + "dpr_recall_idx": recall_idx,
    }


def get_new_dataset(ds, args):
    if args.preflight:
        ds = ds.select(range(10))

    print("=" * 80)
    print("RAG disabled (DPR removed). Scoring positives only.")
    print("=" * 80)

    # score positives
    ds = ds.map(lambda example: get_score(example, rag=False), num_proc=1)

    ds.save_to_disk(str(args.path))
    return ds


def get_roc_auc(positives, negatives):
    from sklearn import metrics

    probs = list(positives) + list(negatives)
    preds = [1] * len(positives) + [0] * len(negatives)
    fpr, tpr, _ = metrics.roc_curve(preds, probs)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
