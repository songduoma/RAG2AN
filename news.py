from pathlib import Path
import datasets
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

FAISS_INDEX_PATH = Path("local/news-please/faiss_index")


class DPR:
    def __init__(self):
        self.ds = datasets.load_dataset("sanxing/advfake_news_please")["train"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 先建 FAISS index（如果不存在就會順便算 embedding）
        self.index_dpr()

        # 問句 encoder（只載一次）
        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        ).to(self.device)
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.q_encoder.eval()

    @torch.no_grad()
    def index_dpr(self):
        """為 news-please dataset 建立 DPR context embedding + FAISS index"""

        def _safe_text(val):
            """Normalize description to a string for tokenization."""
            if isinstance(val, str):
                return val
            if val is None:
                return ""
            if isinstance(val, (list, tuple)):
                return " ".join([x for x in val if isinstance(x, str)])
            return str(val)

        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        faiss_path = FAISS_INDEX_PATH / "my_index.faiss"

        if faiss_path.exists():
            print("loading faiss index")
            # 如果已經有 index，只要把它掛回來
            self.ds.load_faiss_index("embeddings", str(faiss_path))
            return

        print("building DPR context embeddings...")

        ctx_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(self.device)
        ctx_encoder.eval()
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

        def _embed_batch(batch):
            # 取出描述欄並做長度裁剪，避免超過 position embedding 上限
            desc_list = batch.get("description")
            if desc_list is None:
                desc_list = ["" for _ in range(len(next(iter(batch.values()), [])))]
            texts = [_safe_text(t) for t in desc_list]
            inputs = ctx_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # DPR/BERT position embedding limit
            ).to(self.device)

            outputs = ctx_encoder(**inputs)
            # 建議使用 pooler_output（句子向量）
            embeddings = outputs.pooler_output.cpu().numpy()
            return {"embeddings": embeddings}

        ds_with_embeddings = self.ds.map(
            _embed_batch,
            batched=True,
            batch_size=64,
        )

        ds_with_embeddings.add_faiss_index(column="embeddings")

        print("saving faiss index")
        self.ds = ds_with_embeddings
        self.ds.save_faiss_index("embeddings", str(faiss_path))

    def interactive(self):
        """簡單 CLI 測試用"""
        while True:
            try:
                question = input("Enter a question: ")
            except (EOFError, KeyboardInterrupt):
                print("\nexit.")
                break

            if question.strip().lower() == "exit":
                break

            question = question.strip()
            if not question:
                continue

            scores, retrieved_examples = self.search(question)
            for idx, (score, example) in enumerate(zip(scores, retrieved_examples)):
                print(f"{idx} - {score:.2f}")
                print(f'{example.get("date_publish")} - {example.get("url")}')
                print(f'{example.get("description", "")}\n')
            print("---")

    @torch.no_grad()
    def search(self, question, k: int = 10):
        """用 DPR Question Encoder 做最近鄰檢索"""
        enc = self.q_tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(self.device)

        outputs = self.q_encoder(**enc)
        question_embedding = outputs.pooler_output[0].cpu().numpy()

        scores, retrieved_examples = self.ds.get_nearest_examples(
            "embeddings", question_embedding, k=k
        )

        # 把 column-wise dict 轉成 row-wise list[dict]
        retrieved_examples = [
            dict(zip(retrieved_examples, row))
            for row in zip(*retrieved_examples.values())
        ]

        return scores, retrieved_examples


if __name__ == "__main__":
    dpr = DPR()
    dpr.interactive()
