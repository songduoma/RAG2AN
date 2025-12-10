import os
from functools import lru_cache

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)

from search import get_google_ctx

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

        # 加入 output_attentions=True
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_attentions=True,
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

        outputs = self.model(**inputs)

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
        self, text: str, rag_context: str = "", top_k: int = 5
    ) -> dict:
        """
        提供更詳細的分析結果，用於強化 Generator 的 feedback。

        Returns:
            dict: {
                "prob_true": float,           # 被判為真的機率
                "prob_fake": float,           # 被判為假的機率
                "confidence": str,            # 置信度等級 (high/medium/low)
                "suspicious_words": list,     # [(word, score), ...]
                "top_suspicious": str,        # 最可疑的詞
                "detection_reason": str,      # 被偵測到的主要原因
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

        outputs = self.model(**inputs)

        # 計算機率
        probs = torch.softmax(outputs.logits, dim=-1)
        prob_true = probs[0, self.positive_label_id].item()
        prob_fake = 1 - prob_true

        # 判斷置信度
        if prob_fake > 0.8:
            confidence = "high"
        elif prob_fake > 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        # 抓取 attention 分析可疑詞
        attentions = outputs.attentions[-1].cpu()
        avg_att = torch.mean(attentions, dim=1).squeeze(0)
        cls_att = avg_att[0, :]

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
        cls_att[special_mask] = 0

        values, indices = torch.topk(cls_att, k=min(top_k, len(cls_att)))

        suspicious_words = []
        for idx, val in zip(indices, values):
            word = self.tokenizer.decode([input_ids[idx]]).strip()
            if word:  # 過濾空字串
                suspicious_words.append((word, val.item()))

        top_suspicious = suspicious_words[0][0] if suspicious_words else ""

        # 分析偵測原因
        detection_reasons = []
        word_texts = [w[0].lower() for w in suspicious_words]

        # 檢查常見假新聞特徵
        sensational_words = [
            "shocking",
            "unbelievable",
            "secret",
            "exposed",
            "bombshell",
            "stunning",
        ]
        vague_sources = [
            "sources say",
            "reportedly",
            "allegedly",
            "anonymous",
            "insiders",
        ]

        for word in word_texts:
            if any(s in word for s in sensational_words):
                detection_reasons.append("sensationalist_language")
                break

        for word in word_texts:
            if any(s in word for s in vague_sources):
                detection_reasons.append("vague_attribution")
                break

        if prob_fake > 0.7 and not detection_reasons:
            detection_reasons.append("factual_inconsistency")

        if not detection_reasons:
            detection_reasons.append("style_mismatch")

        # 生成改進建議
        suggestions = []

        if "sensationalist_language" in detection_reasons:
            suggestions.append(
                f"Replace sensationalist words like '{top_suspicious}' with neutral alternatives"
            )

        if "vague_attribution" in detection_reasons:
            suggestions.append(
                "Use specific, named sources instead of vague attributions"
            )

        if "factual_inconsistency" in detection_reasons:
            suggestions.append(
                "Ensure all facts align with the provided background context"
            )

        if "style_mismatch" in detection_reasons:
            suggestions.append(
                "Match the formal, objective tone of professional journalism"
            )

        # 通用建議
        suggestions.append(
            f"Avoid or rephrase these flagged terms: {', '.join(word_texts[:3])}"
        )

        if rag_context:
            suggestions.append(
                "Cross-reference your claims with the RAG context provided"
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


def get_retrieval_ctx(example, prefix, source="dpr"):
    """
    Fetch retrieval context for an example. Google search is the default path.
    """
    if source == "none":
        return ""

    if source == "google":
        text = "Related news stories from search results:\n\n"
        query = example.get(prefix + "title") or str(
            example.get(prefix + "description", "")
        )[:50]
        google_ctx = get_google_ctx(query)
        return text + google_ctx + "\n\n" if google_ctx else ""

    if source == "dpr":
        cnt = 0
        text = "Related news stories from search results:\n\n"
        key = prefix + "dpr_retrieved_examples"
        if key not in example:
            return text
        for rex in example[key]:
            if rex["url"] == example["url"]:
                # skip the example itself
                continue
            text += f"{rex['date_publish'].date()} - {rex['title']}\n{rex['url']}\n{rex['description']}\n\n"
            cnt += 1
            if cnt == 5:
                break
        return text

    raise ValueError("Invalid source")


def format_discriminator_input(
    example, rag, prefix="", description_override=None, source="dpr"
):
    """
    Build the discriminator input text with optional RAG context and content override.
    """
    text = ""
    if rag:
        text += get_retrieval_ctx(example, prefix, source=source)
    if text and not text.endswith("\n\n"):
        text += "\n\n"

    title = example.get(prefix + "title", "")
    description = (
        description_override
        if description_override is not None
        else example.get(prefix + "description", "")
    )

    raw_date = example.get(prefix + "date_publish")
    if hasattr(raw_date, "date"):
        date_str = str(raw_date.date())
    else:
        date_str = str(raw_date) if raw_date is not None else "unknown-date"

    text += "Predict the plausibility of the following news story:\n\n"
    text += f"{date_str} - {title}\n{description}\n\n"
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


def get_dpr_results(example, dpr, search_key="title", prefix=""):
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
    ds = ds.map(lambda example: get_score(example, rag=False), num_proc=args.num_proc)

    ds.save_to_disk(str(args.path))
    return ds


def get_roc_auc(positives, negatives):
    from sklearn import metrics

    probs = list(positives) + list(negatives)
    preds = [1] * len(positives) + [0] * len(negatives)
    fpr, tpr, _ = metrics.roc_curve(preds, probs)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
