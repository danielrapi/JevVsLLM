from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


GLUE_SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
AG_NEWS_TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)
AG_NEWS_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)
TREC_TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
TREC_TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
SUPERGLUE_BOOLQ_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
ARC_URL = "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"
HELLASWAG_VAL_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
COMMONSENSEQA_TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
COMMONSENSEQA_DEV_URL = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
RACE_URL = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
BANKING77_TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)


def download_sst2(output: Path, split: str = "dev", limit: int | None = None) -> int:
    """Download GLUE SST-2 and convert it to the local MCQ JSONL format."""

    if split not in {"train", "dev"}:
        raise ValueError("SST-2 only exposes labeled train/dev splits in the GLUE download.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(GLUE_SST2_URL, timeout=60) as response:
        archive_bytes = response.read()

    filename = f"SST-2/{split}.tsv"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open(filename) as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8")
            rows = csv.DictReader(text_file, delimiter="\t")

            count = 0
            with output.open("w", encoding="utf-8") as out_file:
                for row in rows:
                    if limit is not None and count >= limit:
                        break
                    sentence = row["sentence"].strip()
                    label = int(row["label"])
                    record = {
                        "id": f"sst2_{split}_{count:05d}",
                        "question": f"Classify the sentiment of this sentence: {sentence}",
                        "options": ["negative", "positive"],
                        "answer": "A" if label == 0 else "B",
                        "metadata": {
                            "benchmark": "glue_sst2",
                            "split": split,
                            "source": GLUE_SST2_URL,
                            "source_index": count,
                        },
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    return count


def download_ag_news(output: Path, split: str = "test", limit: int | None = None) -> int:
    """Download AG News CSV and convert it to the local MCQ JSONL format."""

    if split not in {"train", "test"}:
        raise ValueError("AG News split must be train or test.")

    labels = {
        "1": "World",
        "2": "Sports",
        "3": "Business",
        "4": "Sci/Tech",
    }
    options = [labels[str(i)] for i in range(1, 5)]
    output.parent.mkdir(parents=True, exist_ok=True)

    source = AG_NEWS_TRAIN_URL if split == "train" else AG_NEWS_TEST_URL
    with urlopen(source, timeout=60) as response:
        text = response.read().decode("utf-8")

    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for row in csv.reader(io.StringIO(text)):
            if limit is not None and count >= limit:
                break
            label, title, description = row
            record = {
                "id": f"ag_news_test_{count:05d}",
                "question": (
                    "Classify the topic of this news item.\n\n"
                    f"Title: {title}\nDescription: {description}"
                ),
                "options": options,
                "answer": int(label) - 1,
                "metadata": {
                    "benchmark": "ag_news",
                    "split": split,
                    "source": source,
                    "source_index": count,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_trec(output: Path, split: str = "test", limit: int | None = None) -> int:
    """Download TREC question classification data and convert to MCQ JSONL."""

    if split not in {"train", "test"}:
        raise ValueError("TREC split must be train or test.")

    options = ["abbreviation", "description", "entity", "human", "location", "number"]
    coarse_to_index = {
        "ABBR": 0,
        "DESC": 1,
        "ENTY": 2,
        "HUM": 3,
        "LOC": 4,
        "NUM": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    source = TREC_TRAIN_URL if split == "train" else TREC_TEST_URL
    with urlopen(source, timeout=60) as response:
        text = response.read().decode("latin-1")

    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for line in text.splitlines():
            if limit is not None and count >= limit:
                break
            if not line.strip():
                continue
            label, question = line.split(" ", 1)
            coarse = label.split(":", 1)[0]
            record = {
                "id": f"trec_test_{count:05d}",
                "question": f"Classify the type of this question: {question.strip()}",
                "options": options,
                "answer": coarse_to_index[coarse],
                "metadata": {
                    "benchmark": "trec_question_classification",
                    "split": split,
                    "source": source,
                    "source_index": count,
                    "fine_label": label,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_boolq(output: Path, split: str = "val", limit: int | None = None) -> int:
    """Download SuperGLUE BoolQ data and convert to binary MCQ JSONL."""

    if split not in {"train", "val"}:
        raise ValueError("BoolQ split must be train or val.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SUPERGLUE_BOOLQ_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = _find_zip_member(archive, f"BoolQ/{split}.jsonl")
        count = 0
        with archive.open(name) as raw_file, output.open("w", encoding="utf-8") as out_file:
            for raw_line in raw_file:
                if limit is not None and count >= limit:
                    break
                row = json.loads(raw_line.decode("utf-8"))
                record = {
                    "id": f"boolq_val_{count:05d}",
                    "question": (
                        "Answer the yes/no question using the passage.\n\n"
                        f"Passage: {row['passage']}\n\nQuestion: {row['question']}"
                    ),
                    "options": ["no", "yes"],
                    "answer": 1 if row["label"] else 0,
                    "metadata": {
                        "benchmark": "superglue_boolq",
                        "split": "train" if split == "train" else "validation",
                        "source": SUPERGLUE_BOOLQ_URL,
                        "source_index": row.get("idx", count),
                    },
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def download_arc_challenge(output: Path, split: str = "Dev", limit: int | None = None) -> int:
    """Download AI2 ARC-Challenge and convert to MCQ JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ARC_URL, timeout=60) as response:
        archive_bytes = response.read()

    suffix = f"ARC-Challenge/ARC-Challenge-{split}.jsonl"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = _find_zip_member(archive, suffix)
        count = 0
        with archive.open(name) as raw_file, output.open("w", encoding="utf-8") as out_file:
            for raw_line in raw_file:
                if limit is not None and count >= limit:
                    break
                row = json.loads(raw_line.decode("utf-8"))
                choices = row["question"]["choices"]
                labels = [choice["label"] for choice in choices]
                options = [choice["text"] for choice in choices]
                answer_label = row["answerKey"]
                if answer_label not in labels:
                    continue
                record = {
                    "id": f"arc_challenge_{split.lower()}_{count:05d}",
                    "question": row["question"]["stem"],
                    "options": options,
                    "answer": labels.index(answer_label),
                    "metadata": {
                        "benchmark": "arc_challenge",
                        "split": split.lower(),
                        "source": ARC_URL,
                        "source_id": row.get("id"),
                        "source_index": count,
                        "source_choice_labels": labels,
                    },
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def download_arc_challenge_combined(output: Path, limit: int | None = None) -> int:
    """Download all labeled ARC-Challenge splits and convert to MCQ JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ARC_URL, timeout=60) as response:
        archive_bytes = response.read()

    count = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive, output.open(
        "w", encoding="utf-8"
    ) as out_file:
        for split in ("Train", "Dev", "Test"):
            suffix = f"ARC-Challenge/ARC-Challenge-{split}.jsonl"
            name = _find_zip_member(archive, suffix)
            with archive.open(name) as raw_file:
                for raw_line in raw_file:
                    if limit is not None and count >= limit:
                        return count
                    row = json.loads(raw_line.decode("utf-8"))
                    choices = row["question"]["choices"]
                    labels = [choice["label"] for choice in choices]
                    options = [choice["text"] for choice in choices]
                    answer_label = row["answerKey"]
                    if answer_label not in labels:
                        continue
                    record = {
                        "id": f"arc_challenge_all_{count:05d}",
                        "question": row["question"]["stem"],
                        "options": options,
                        "answer": labels.index(answer_label),
                        "metadata": {
                            "benchmark": "arc_challenge",
                            "split": split.lower(),
                            "source": ARC_URL,
                            "source_id": row.get("id"),
                            "source_index": count,
                            "source_choice_labels": labels,
                        },
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def download_20newsgroups(
    output: Path, split: str = "test", limit: int | None = None
) -> int:
    """Download 20 Newsgroups through sklearn and convert to MCQ JSONL."""

    if split not in {"train", "test", "all"}:
        raise ValueError("20 Newsgroups split must be train, test, or all.")

    from sklearn.datasets import fetch_20newsgroups

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_20newsgroups(
        subset=split,
        remove=("headers", "footers", "quotes"),
    )
    options = list(dataset.target_names)
    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for text, target in zip(dataset.data, dataset.target, strict=True):
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            if limit is not None and count >= limit:
                break
            record = {
                "id": f"twenty_newsgroups_test_{count:05d}",
                "question": f"Classify this newsgroup post:\n\n{cleaned[:4000]}",
                "options": options,
                "answer": int(target),
                "metadata": {
                    "benchmark": "20_newsgroups",
                    "split": split,
                    "source": "sklearn.datasets.fetch_20newsgroups",
                    "source_index": count,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_hellaswag(output: Path, limit: int | None = None) -> int:
    """Download HellaSwag validation and convert to MCQ JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(HELLASWAG_VAL_URL, timeout=60) as response:
        text = response.read().decode("utf-8")

    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for line in text.splitlines():
            if limit is not None and count >= limit:
                break
            row = json.loads(line)
            context = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
            endings = row["endings"]
            label = int(row["label"])
            record = {
                "id": f"hellaswag_val_{count:05d}",
                "question": (
                    "Choose the most plausible ending for the situation.\n\n"
                    f"Context: {context}"
                ),
                "options": endings,
                "answer": label,
                "metadata": {
                    "benchmark": "hellaswag",
                    "split": "validation",
                    "source": HELLASWAG_VAL_URL,
                    "source_index": count,
                    "activity_label": row.get("activity_label"),
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_commonsenseqa(
    output: Path, split: str = "train", limit: int | None = None
) -> int:
    """Download CommonsenseQA and convert to MCQ JSONL."""

    if split not in {"train", "dev"}:
        raise ValueError("CommonsenseQA split must be train or dev.")

    source = COMMONSENSEQA_TRAIN_URL if split == "train" else COMMONSENSEQA_DEV_URL
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source, timeout=60) as response:
        text = response.read().decode("utf-8")

    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for line in text.splitlines():
            if limit is not None and count >= limit:
                break
            row = json.loads(line)
            choices = row["question"]["choices"]
            labels = [choice["label"] for choice in choices]
            options = [choice["text"] for choice in choices]
            answer_key = row["answerKey"]
            if answer_key not in labels:
                continue
            record = {
                "id": f"commonsenseqa_{split}_{count:05d}",
                "question": row["question"]["stem"],
                "options": options,
                "answer": labels.index(answer_key),
                "metadata": {
                    "benchmark": "commonsenseqa",
                    "split": split,
                    "source": source,
                    "source_id": row.get("id"),
                    "source_index": count,
                    "source_choice_labels": labels,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_mmlu(output: Path, split: str = "test", limit: int | None = None) -> int:
    """Download MMLU and convert subject CSVs to MCQ JSONL."""

    if split not in {"dev", "val", "test"}:
        raise ValueError("MMLU split must be dev, val, or test.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MMLU_URL, timeout=120) as response:
        archive_bytes = response.read()

    count = 0
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive, output.open(
        "w", encoding="utf-8"
    ) as out_file:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile()
                and f"/{split}/" in member.name
                and member.name.endswith(f"_{split}.csv")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            if limit is not None and count >= limit:
                break
            subject = Path(member.name).stem.removesuffix(f"_{split}").replace("_", " ")
            raw = archive.extractfile(member)
            if raw is None:
                continue
            text_file = io.TextIOWrapper(raw, encoding="utf-8")
            for row in csv.reader(text_file):
                if limit is not None and count >= limit:
                    break
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = row[:6]
                answer = answer.strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                record = {
                    "id": f"mmlu_{split}_{count:05d}",
                    "question": f"Subject: {subject}\n\n{question}",
                    "options": [a, b, c, d],
                    "answer": answer,
                    "metadata": {
                        "benchmark": "mmlu",
                        "split": split,
                        "source": MMLU_URL,
                        "source_file": member.name,
                        "subject": subject,
                        "source_index": count,
                    },
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def download_race(output: Path, split: str = "train", limit: int | None = None) -> int:
    """Download RACE and convert reading-comprehension questions to MCQ JSONL."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("RACE split must be train, dev, or test.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(RACE_URL, timeout=120) as response:
        archive_bytes = response.read()

    count = 0
    split_dir = f"RACE/{split}/"
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive, output.open(
        "w", encoding="utf-8"
    ) as out_file:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.isfile() and split_dir in member.name and member.name.endswith(".txt")
            ),
            key=lambda member: member.name,
        )
        for member in members:
            if limit is not None and count >= limit:
                break
            raw = archive.extractfile(member)
            if raw is None:
                continue
            row = json.loads(raw.read().decode("utf-8"))
            article = " ".join(row["article"].split())
            for question, options, answer in zip(
                row["questions"], row["options"], row["answers"], strict=True
            ):
                if limit is not None and count >= limit:
                    break
                answer = answer.strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                record = {
                    "id": f"race_{split}_{count:05d}",
                    "question": (
                        "Answer the reading comprehension question using the passage.\n\n"
                        f"Passage: {article[:5000]}\n\nQuestion: {question}"
                    ),
                    "options": options,
                    "answer": answer,
                    "metadata": {
                        "benchmark": "race",
                        "split": split,
                        "source": RACE_URL,
                        "source_file": member.name,
                        "source_id": row.get("id"),
                        "source_index": count,
                    },
                }
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def download_banking77(output: Path, split: str = "test", limit: int | None = None) -> int:
    """Download BANKING77 and convert to fine-grained intent classification JSONL."""

    if split not in {"train", "test"}:
        raise ValueError("BANKING77 split must be train or test.")

    source = BANKING77_TRAIN_URL if split == "train" else BANKING77_TEST_URL
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source, timeout=60) as response:
        text = response.read().decode("utf-8")

    rows = list(csv.DictReader(io.StringIO(text)))
    options = sorted({row["category"].strip() for row in rows})
    option_to_index = {option: index for index, option in enumerate(options)}

    count = 0
    with output.open("w", encoding="utf-8") as out_file:
        for row in rows:
            if limit is not None and count >= limit:
                break
            category = row["category"].strip()
            record = {
                "id": f"banking77_{split}_{count:05d}",
                "question": (
                    "Classify the banking customer support intent for this message.\n\n"
                    f"Message: {row['text'].strip()}"
                ),
                "options": options,
                "answer": option_to_index[category],
                "metadata": {
                    "benchmark": "banking77",
                    "split": split,
                    "source": source,
                    "source_index": count,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_starter_suite(output_dir: Path, limit: int = 100) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sst2": download_sst2(output_dir / "sst2_dev_100.jsonl", split="dev", limit=limit),
        "ag_news": download_ag_news(output_dir / "ag_news_test_100.jsonl", limit=limit),
        "trec": download_trec(output_dir / "trec_test_100.jsonl", limit=limit),
        "boolq": download_boolq(output_dir / "boolq_val_100.jsonl", limit=limit),
        "arc_challenge": download_arc_challenge(
            output_dir / "arc_challenge_dev_100.jsonl", split="Dev", limit=limit
        ),
        "20_newsgroups": download_20newsgroups(
            output_dir / "20_newsgroups_test_100.jsonl", limit=limit
        ),
    }


def download_large_hard_suite(output_dir: Path, limit: int = 2000) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sst2_train": download_sst2(
            output_dir / f"sst2_train_{limit}.jsonl", split="train", limit=limit
        ),
        "ag_news_train": download_ag_news(
            output_dir / f"ag_news_train_{limit}.jsonl", split="train", limit=limit
        ),
        "trec_train": download_trec(
            output_dir / f"trec_train_{limit}.jsonl", split="train", limit=limit
        ),
        "boolq_train": download_boolq(
            output_dir / f"boolq_train_{limit}.jsonl", split="train", limit=limit
        ),
        "arc_challenge_all": download_arc_challenge_combined(
            output_dir / f"arc_challenge_all_{limit}.jsonl", limit=limit
        ),
        "20_newsgroups_train": download_20newsgroups(
            output_dir / f"20_newsgroups_train_{limit}.jsonl", split="train", limit=limit
        ),
        "race_train": download_race(
            output_dir / f"race_train_{limit}.jsonl", split="train", limit=limit
        ),
        "commonsenseqa_train": download_commonsenseqa(
            output_dir / f"commonsenseqa_train_{limit}.jsonl", split="train", limit=limit
        ),
        "mmlu_test": download_mmlu(
            output_dir / f"mmlu_test_{limit}.jsonl", split="test", limit=limit
        ),
    }


def download_focused_suite(output_dir: Path, limit: int = 2000) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "banking77_test": download_banking77(
            output_dir / f"banking77_test_{limit}.jsonl", split="test", limit=limit
        ),
        "ag_news_train": download_ag_news(
            output_dir / f"ag_news_train_{limit}.jsonl", split="train", limit=limit
        ),
        "boolq_train": download_boolq(
            output_dir / f"boolq_train_{limit}.jsonl", split="train", limit=limit
        ),
        "arc_challenge_all": download_arc_challenge_combined(
            output_dir / f"arc_challenge_all_{limit}.jsonl", limit=limit
        ),
        "mmlu_test": download_mmlu(
            output_dir / f"mmlu_test_{limit}.jsonl", split="test", limit=limit
        ),
    }


def _find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    for name in archive.namelist():
        if name.endswith(suffix):
            return name
    raise ValueError(f"Could not find {suffix} in archive.")
