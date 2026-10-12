from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen


GLUE_SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
AG_NEWS_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)
TREC_TEST_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
SUPERGLUE_BOOLQ_URL = "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip"
ARC_URL = "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"


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


def download_ag_news(output: Path, limit: int | None = None) -> int:
    """Download AG News test CSV and convert it to the local MCQ JSONL format."""

    labels = {
        "1": "World",
        "2": "Sports",
        "3": "Business",
        "4": "Sci/Tech",
    }
    options = [labels[str(i)] for i in range(1, 5)]
    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(AG_NEWS_TEST_URL, timeout=60) as response:
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
                    "split": "test",
                    "source": AG_NEWS_TEST_URL,
                    "source_index": count,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_trec(output: Path, limit: int | None = None) -> int:
    """Download TREC question classification test data and convert to MCQ JSONL."""

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

    with urlopen(TREC_TEST_URL, timeout=60) as response:
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
                    "split": "test",
                    "source": TREC_TEST_URL,
                    "source_index": count,
                    "fine_label": label,
                },
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def download_boolq(output: Path, limit: int | None = None) -> int:
    """Download SuperGLUE BoolQ validation data and convert to binary MCQ JSONL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SUPERGLUE_BOOLQ_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        name = _find_zip_member(archive, "BoolQ/val.jsonl")
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
                        "split": "validation",
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


def download_20newsgroups(output: Path, limit: int | None = None) -> int:
    """Download 20 Newsgroups test split through sklearn and convert to MCQ JSONL."""

    from sklearn.datasets import fetch_20newsgroups

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_20newsgroups(
        subset="test",
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
                    "split": "test",
                    "source": "sklearn.datasets.fetch_20newsgroups",
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


def _find_zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    for name in archive.namelist():
        if name.endswith(suffix):
            return name
    raise ValueError(f"Could not find {suffix} in archive.")
