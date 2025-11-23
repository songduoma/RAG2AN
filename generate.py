from pathlib import Path
import argparse
import os
from functools import lru_cache

import datasets
import torch
from thefuzz import fuzz
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)

from news import DPR
from utils import get_gpt_response
from search import get_google_ctx

logging.set_verbosity_error() 


DATASET_PATH = "local/hf_datasets/"
GPT4 = "gpt-4o-2024-05-13"
REWRITE_THRESHOLD = 60
DEFAULT_ENCODER_MODEL = os.environ.get(
    "ENCODER_DISCRIMINATOR_MODEL", "microsoft/deberta-v3-base-mnli"
)
MAX_ENCODER_SEQ_LEN = int(os.environ.get("ENCODER_DISCRIMINATOR_MAX_LEN", "512"))


class EncoderDiscriminator:
    """
    Lightweight encoder-based discriminator that predicts the plausibility
    of a news story using a sequence classification model.
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
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.positive_label_id = self._detect_positive_label_id()

    def _detect_positive_label_id(self) -> int:
        """
        Try to infer which label corresponds to 'real/entail/true' so that the
        discriminator can output a plausibility probability.
        """
        id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        for idx, label in id2label.items():
            label_lower = label.lower()
            if any(
                key in label_lower
                for key in ("true", "real", "entail", "support", "pos", "positive")
            ):
                return int(idx)
        # default to label 1 if present, otherwise 0
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


@lru_cache(maxsize=1)
def get_encoder_discriminator(model_name: str = DEFAULT_ENCODER_MODEL) -> EncoderDiscriminator:
    return EncoderDiscriminator(model_name=model_name)


def get_retrieval_ctx(example, prefix, source="dpr"):
    cnt = 0
    text = "Related news stories from search results:\n\n"
    if source == 'google':
        text += get_google_ctx(example[prefix + "title"]) + "\n\n"
    elif source == 'dpr':
        for rex in example[prefix + "dpr_retrieved_examples"]:
            if rex["url"] == example["url"]:
                # skip the example itself
                continue
            text += f'{rex["date_publish"].date()} - {rex["title"]}\n{rex["url"]}\n{rex["description"]}\n\n'
            cnt += 1
            if cnt == 5:
                break
    else:
        raise ValueError("Invalid source")
    return text


def get_score(example, rag, prefix="", rationale=False, model=None):
    """
    Score the plausibility of a news story using an encoder-based discriminator.

    Args:
        example: The example to get the score for.
        rag: Whether to use RAG context.
        prefix: Empty string or "f_" for fake news.
        rationale: Whether to ask for rationale (Chain of Thought).
        model: Deprecated; kept for API compatibility.
    Returns:
        A dict with the plausibility score (1-10), raw predictions list,
        variance (0 for deterministic encoder), and a majority flag.
    """

    text = get_retrieval_ctx(example, prefix) if rag else ""
    text += "Predict the plausibility of the following news story:\n\n"
    text += f'{example["date_publish"].date()} - {example[prefix + "title"]}\n{example[prefix + "description"]}\n\n'

    discriminator = get_encoder_discriminator()
    prob_true = discriminator.predict_prob(text)

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
    }


def get_rationale(example, rag, prefix=""):

    text = get_retrieval_ctx(example, prefix) if rag else ""
    text += "Please predict the plausibility of the following news story:\n\n"
    text += f'{example["date_publish"].date()} - {example[prefix + "title"]}\n{example[prefix + "description"]}\n\n'

    res = get_gpt_response(
        messages=[
            {
                "role": "system",
                "content": "Today is March 26, 2024. You fact-check a news you haven't seen, given a list of related news stories from search results.",
            },
            {
                "role": "user",
                "content": text
                + "\n\nPlease explain what you find suspicious about the news story. Give the top 3 points in a list format and nothing else.",
            },
        ],
        model=GPT4,
        temperature=0,
    )

    rationale = res.choices[0].message.content.strip()

    return {
        prefix + ("rag_" if rag else "") + "rationale": rationale,
    }


def confirm_fake(title, description, f_title, f_description):

    title_similarity = fuzz.ratio(f_title, title)
    description_similarity = fuzz.ratio(f_description, description)
    if (
        title_similarity <= REWRITE_THRESHOLD
        or description_similarity <= REWRITE_THRESHOLD
    ):
        return False, [
            "Fail to satisfy rewrite similarity threshold",
            title_similarity,
            description_similarity,
        ]

    text = f"News 1\nTitle: {title}\nDescription: {description}\n\nNews 2\nTitle: {f_title}\nDescription: {f_description}"

    res = get_gpt_response(
        messages=[
            {
                "role": "user",
                "content": text
                + "\n\nDoes News 2 conflict with News 1? Please type yes or no and nothing else.",
            },
        ],
        model=GPT4,
        temperature=0.0,
        n=10,
    )

    raw_output = [c.message.content.lower() for c in res.choices]
    ratio = sum(["yes" in c.message.content.lower() for c in res.choices]) / len(
        res.choices
    )

    return ratio >= 0.8, raw_output


def adv_rewrite(example, rag=False):

    text = "Please rewrite the following news story.\n\n"
    # date - title - description
    text += f'News Date: {example["date_publish"].date()}\nTitle: {example["title"]}\nDescription: {example["description"]}\n\n'
    text += f'It was previously rewrote as:\nTitle: {example["f_title"]}\nDescription: {example["f_description"]}\n\n'
    rationale = example["f_rag_rationale"] if rag else example["f_rationale"]
    text += f'Fact-checkers found the following suspicious:\n{rationale}\n\n'
    text += f"Please rewrite the news story to make it more believable and fool the fact-checkers."

    date_instructions = "Today is March 26, 2024."
    open_instructions = " Please rewrite a real recent news story to introduce some believable factual errors. The modification should be able to fool professional fact-checkers."
    format_instructions = "\n\nFormat\nError introduced:\nTitle:\nDescription:"

    instructions = date_instructions + open_instructions + format_instructions

    samples = []
    error_message = ["Failed to rewrite:", text]

    for _ in range(5):
        res = get_gpt_response(
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {"role": "user", "content": text},
            ],
            model=GPT4,
            temperature=1.0,
            n=8,
            stop=["Fact-checkers"],
            max_tokens=300,
        )

        for i in range(len(res.choices)):
            content = res.choices[i].message.content
            error_message.append(content)
            # try parse the title and description, otherwise return empty strings
            try:
                content = content.replace(":\n", ": ")
                items = content.strip().split("\n")
                items = [
                    item for item in items if item and not item.startswith("News Date:")
                ]
                error, title, description = items[:3]
                title = title[7:] if title.startswith("Title: ") else title
                description = (
                    description[13:]
                    if description.startswith("Description: ")
                    else description
                )
                error = error[18:] if error.startswith("Error introduced: ") else error

                success, raw_output = confirm_fake(
                    example["title"], example["description"], title, description
                )
                if success:
                    samples.append(
                        (
                            get_score(example, True, "f_", rationale=False)[
                                "f_rag_score"
                            ],
                            {
                                "f_title": title,
                                "f_description": description,
                                "f_error": error,
                            },
                        )
                    )
                else:
                    error_message.append(str(raw_output))
            except Exception as e:
                error_message.append(str(e))
                continue

        if len(samples) > 0:
            break

    if len(samples) == 0:
        return {
            "f_title": "",
            "f_description": "",
            "f_error": "",
        }
    else:
        # return the one with the highest score
        return max(samples, key=lambda x: x[0])[1]


def rewrite(example, rag=False, type="entity"):

    text = (
        (
            get_retrieval_ctx(example, prefix="f_")
            + f"Please rewrite the following news story.\n\n"
        )
        if rag
        else ""
    )
    # date - title - description
    text += f'News Date: {example["date_publish"].date()}\nTitle: {example["title"]}\nDescription: {example["description"]}'

    date_instructions = "Today is March 26, 2024."
    subs_instructions = " You will be given a recent news story, please rewrite it by substituting one or two entities (names or locations) to their equivalence, but the news should still looks true."
    open_instructions = " Please rewrite a real recent news story to introduce some believable factual errors. The modification should be able to fool professional fact-checkers."
    rag_instructions = (
        " You should also consider a list of related news stories from search results that people might use to fact-check the news story you write."
        if rag
        else ""
    )
    format_instructions = "\n\nFormat\nError introduced:\nTitle:\nDescription:"

    instructions = (
        date_instructions
        + (subs_instructions if type == "entity" else open_instructions)
        + rag_instructions
        + format_instructions
    )

    for _ in range(5):
        res = get_gpt_response(
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {"role": "user", "content": text},
            ],
            model=GPT4,
            temperature=1.0,
            n=5,
        )

        for i in range(len(res.choices)):
            content = res.choices[i].message.content
            # try parse the title and description, otherwise return empty strings
            try:
                content = content.replace(":\n", ": ")
                items = content.strip().split("\n")
                items = [item for item in items if item]
                error, title, description = items
                title = title[7:] if title.startswith("Title: ") else title
                description = (
                    description[13:]
                    if description.startswith("Description: ")
                    else description
                )
                error = error[18:] if error.startswith("Error introduced: ") else error
                if confirm_fake(
                    example["title"], example["description"], title, description
                ):
                    return {
                        "f_title": title,
                        "f_description": description,
                        "f_error": error,
                    }
            except:
                continue

    print("Failed to rewrite")
    return {
        "f_title": "",
        "f_description": "",
        "f_error": "",
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
        # get the first n examples
        n = 10
        ds = ds.select(range(n))
        print("Preflight check with {n} examples".format(n=n))

    print("=" * 80)
    print("Initiating RAG")
    dpr = DPR()
    print("=" * 80)

    if args.first_round:
        # get score for positive examples
        ds = ds.map(lambda example: get_score(example, rag=False), num_proc=args.num_proc)
        print("Scored positive examples")

        # get DPR results for positive examples, num_proc=1 is important
        ds = ds.map(lambda example: get_dpr_results(example, dpr), num_proc=1)

        # get rag score for positive examples
        ds = ds.map(
            lambda example: get_score(example, rag=True, rationale=False),
            num_proc=args.num_proc,
        )
        print("Scored positive examples w/ RAG")

    shift = 1
    for round_num in range(shift, args.num_rounds + shift):

        print(f"Round {round_num}")

        if args.generation_context_type == "none":
            # generate negative examples (fake news)
            ds = ds.map(
                lambda example: rewrite(
                    example, rag=False, type=args.substitution_type
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples")

        elif args.generation_context_type == "rag_raw":
            # generate negative examples (fake news) w/ DPR
            ds = ds.map(
                lambda example: rewrite(example, rag=True, type=args.substitution_type),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ retrieval context")

        elif args.generation_context_type == "rag_rationale":
            ds = ds.map(
                lambda example: adv_rewrite(
                    example, rag=True
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ detector rationale (RAG)")

        elif args.generation_context_type == "rationale":
            ds = ds.map(
                lambda example: adv_rewrite(
                    example, rag=False
                ),
                num_proc=args.num_proc,
            )
            print("Generated negative examples w/ detector rationale")

        # filter out examples that are not rewritten, i.e., empty f_title or f_description
        size_before_filter = ds.num_rows
        ds = ds.filter(
            lambda example: len(example["f_title"]) > 0
            and len(example["f_description"]) > 0
        )
        print(
            f"Filtered out {size_before_filter - ds.num_rows} examples that could not be rewritten."
        )

        # get score for negative examples
        ds = ds.map(
            lambda example: get_score(example, rag=False, prefix="f_"), num_proc=args.num_proc
        )
        print("Scored negative examples")

        # get DPR results for negative examples
        ds = ds.map(
            lambda example: get_dpr_results(example, dpr, prefix="f_"), num_proc=1
        )

        # get rag score for negative examples
        ds = ds.map(
            lambda example: get_score(example, rag=True, prefix="f_", rationale=False),
            num_proc=args.num_proc,
        )
        print("Scored negative examples w/ RAG")

        # get post-hoc rationale (w/o RAG) for negative examples
        ds = ds.map(
            lambda example: get_rationale(example, False, "f_"), num_proc=args.num_proc
        )
        # get post-hoc rationale (w/ RAG) for negative examples
        ds = ds.map(
            lambda example: get_rationale(example, True, "f_"), num_proc=args.num_proc
        )

        print(f"Round {round_num} completed")
        print(f'ROC AUC: {get_roc_auc(ds["score"], ds["f_score"])}')
        print(f'ROC AUC (RAG): {get_roc_auc(ds["rag_score"], ds["f_rag_score"])}')

        # save the dataset to disk
        ds.save_to_disk(str(args.path) + f"_round{round_num}")
    return ds


def get_roc_auc(positives, negatives):
    from sklearn import metrics

    probs = list(positives) + list(negatives)
    preds = [1] * len(positives) + [0] * len(negatives)
    fpr, tpr, thresholds = metrics.roc_curve(preds, probs)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


if __name__ == "__main__":
    """
    DISCLAIMER: This code is for research purposes only, specifically for studying misinformation detection
    and improving fact-checking systems. Any use of this code for generating and spreading actual 
    misinformation is strictly prohibited and unethical.
    
    RESPONSIBLE USAGE:
    - Use only for academic research and improving detection systems
    - Do not deploy for generating actual fake news
    - Follow ethical guidelines for AI research
    """
    # create a parser
    parser = argparse.ArgumentParser(
        description="Research tool for studying misinformation detection through adversarial examples."
    )
    # add arguments to the parser
    parser.add_argument(
        "--source",
        type=str,
        help="The source dataset to be used.",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="The target dataset to be created.",
        required=True,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Whether to run the preflight check on about 10 examples.",
    )
    parser.add_argument(
        "--first-round",
        action="store_true",
        help="Whether to run the first round of the game. Only in the first round, we score the positives.",
    )
    parser.add_argument(
        "--substitution-type",
        type=str,
        default="open",
        help="The type of generation: entity, open",
    )
    parser.add_argument(
        "--generation-context-type",
        type=str,
        default="none",
        help="The type of context: none, rag_raw, rag_rationale",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="The number of rounds to run the game.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=10, help="The number of processors to use."
    )

    # parse the arguments
    args = parser.parse_args()


    if not args.source:
        ds = datasets.load_dataset('sanxing/advfake')
        # convert the date_publish to timestamp
        import datetime
        ds = ds['train'].map(lambda x: {'date_publish_timestamp': datetime.datetime.strptime(x['date_publish'], '%Y-%m-%d %H:%M:%S')})
        # drop the date_publish column
        ds = ds.remove_columns('date_publish')
        # change the date_publish_timestamp to date_publish
        ds = ds.rename_column('date_publish_timestamp', 'date_publish')

    else:
        source_path = DATASET_PATH + args.source
        # verify path
        if not Path(source_path).exists():
            raise ValueError(f"{DATASET_PATH + args.source} does not exist.")
        ds = datasets.load_from_disk(source_path)

    # if directory exists, give error
    args.path = Path(f"{DATASET_PATH}{args.target}")
    if args.path.exists():
        raise ValueError(f"{args.path} already exists.")

    ds = get_new_dataset(ds, args)
