import json
from pathlib import Path

from datasets import load_from_disk


def inspect_run(run_dir: str, sample_n: int = 3) -> None:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"{run_dir} not found")

    # 1) collect round stats
    rounds = sorted(base.glob("round_*"))
    print(f"Found {len(rounds)} rounds under {run_dir}")
    for rd in rounds:
        state_file = rd / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(f"{rd.name}: {state}")
        else:
            print(f"{rd.name}: missing state.json")

    # 2) inspect last round dataset
    if not rounds:
        return
    last_round = rounds[-1]
    ds = load_from_disk(str(last_round))
    print(f"\nLast round: {last_round.name}")
    print("Columns:", ds.column_names)
    print("Num samples:", len(ds))

    for i in range(min(sample_n, len(ds))):
        row = ds[i]
        print(f"\nSample {i}:")
        print("prob_true:", row.get("prob_true"))
        print("suspicious_words:", row.get("suspicious_words"))
        if "feedback_prompt" in row:
            print("feedback_prompt:", row["feedback_prompt"])
        print("text snippet:\n", row["text"][:400])


if __name__ == "__main__":
    # 修改 run_dir 指向你的輸出資料夾；sample_n 決定要看幾筆
    inspect_run(run_dir="local/rag_gan_runs/20251127_145105", sample_n=3)
