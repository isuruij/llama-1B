import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_DIR = Path("new_data")
DEFAULT_OUTPUT_DIR = Path("new_split_v2")
DEFAULT_SEED = 42
TEST_RATIO_NUMERATOR = 1
TEST_RATIO_DENOMINATOR = 10
REQUIRED_FIELDS = {
    "answer",
    "answerable",
    "chapter",
    "chapter_title",
    "context",
    "grade",
    "question",
}


def normalized_text(value):
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", value).strip()


def context_key(record):
    return normalized_text(record["context"])


def load_records(input_dir):
    records = []
    fingerprints = set()

    input_files = sorted(
        input_dir.glob("*.jsonl"),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else math.inf, path.name),
    )
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir.resolve()}")

    for path in input_files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error

                missing = REQUIRED_FIELDS.difference(record)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing fields {sorted(missing)}"
                    )
                if type(record["answerable"]) is not bool:
                    raise TypeError(
                        f"{path}:{line_number}: answerable must be a JSON boolean"
                    )
                if not isinstance(record["context"], str) or not record["context"].strip():
                    raise ValueError(f"{path}:{line_number}: context is empty")

                fingerprint = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if fingerprint in fingerprints:
                    raise ValueError(f"{path}:{line_number}: exact duplicate record")
                fingerprints.add(fingerprint)
                records.append(record)

    return records, input_files


def allocate_test_targets(records):
    totals = Counter(
        (str(record["grade"]), record["answerable"]) for record in records
    )
    targets = {
        key: count * TEST_RATIO_NUMERATOR // TEST_RATIO_DENOMINATOR
        for key, count in totals.items()
    }
    target_total = math.ceil(
        len(records) * TEST_RATIO_NUMERATOR / TEST_RATIO_DENOMINATOR
    )

    remaining = target_total - sum(targets.values())
    largest_remainders = sorted(
        totals,
        key=lambda key: (
            -(totals[key] * TEST_RATIO_NUMERATOR % TEST_RATIO_DENOMINATOR),
            int(key[0]) if key[0].isdigit() else math.inf,
            not key[1],
        ),
    )
    for key in largest_remainders[:remaining]:
        targets[key] += 1

    for key, total in totals.items():
        if total > 1 and targets[key] == 0:
            raise ValueError(f"Test allocation omitted stratum {key}")
        if targets[key] >= total:
            raise ValueError(f"Train allocation omitted stratum {key}")

    return totals, targets, target_total


def group_records(records):
    groups_by_context = defaultdict(list)
    for record in records:
        groups_by_context[context_key(record)].append(record)

    groups_by_grade = defaultdict(list)
    for context, group in groups_by_context.items():
        grades = {str(record["grade"]) for record in group}
        if len(grades) != 1:
            raise ValueError(
                f"A shared context occurs across multiple grades: {sorted(grades)}"
            )
        groups_by_grade[next(iter(grades))].append((context, group))

    return groups_by_context, groups_by_grade


def select_test_groups(groups_by_grade, targets, seed):
    selected_contexts = set()

    for grade in sorted(
        groups_by_grade,
        key=lambda value: int(value) if value.isdigit() else math.inf,
    ):
        groups = list(groups_by_grade[grade])
        random.Random(f"{seed}:{grade}").shuffle(groups)
        target = (
            targets[(grade, True)],
            targets[(grade, False)],
        )

        states = {(0, 0): ()}
        for index, (context, group) in enumerate(groups):
            contribution = (
                sum(record["answerable"] is True for record in group),
                sum(record["answerable"] is False for record in group),
            )
            additions = {}
            for counts, chosen_indexes in states.items():
                new_counts = (
                    counts[0] + contribution[0],
                    counts[1] + contribution[1],
                )
                if new_counts[0] > target[0] or new_counts[1] > target[1]:
                    continue
                if new_counts not in states and new_counts not in additions:
                    additions[new_counts] = chosen_indexes + (index,)
            states.update(additions)
            if target in states:
                break

        if target not in states:
            raise RuntimeError(
                f"Could not create an exact grouped allocation for grade {grade}: "
                f"target true={target[0]}, false={target[1]}"
            )

        for index in states[target]:
            selected_contexts.add(groups[index][0])

    return selected_contexts


def split_records(records, selected_contexts, seed):
    train_records = []
    test_records = []

    for record in records:
        destination = (
            test_records
            if context_key(record) in selected_contexts
            else train_records
        )
        destination.append(record)

    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(test_records)
    return train_records, test_records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def validate_split(records, train_records, test_records, targets):
    if len(train_records) + len(test_records) != len(records):
        raise AssertionError("Split size does not match the source size")

    train_contexts = {context_key(record) for record in train_records}
    test_contexts = {context_key(record) for record in test_records}
    shared_contexts = train_contexts.intersection(test_contexts)
    if shared_contexts:
        raise AssertionError(
            f"{len(shared_contexts)} contexts occur in both train and test"
        )

    test_counts = Counter(
        (str(record["grade"]), record["answerable"]) for record in test_records
    )
    if test_counts != Counter(targets):
        raise AssertionError(
            f"Test strata differ from targets: {test_counts} != {targets}"
        )

    train_labels = Counter(record["answerable"] for record in train_records)
    test_labels = Counter(record["answerable"] for record in test_records)
    if set(train_labels) != {True, False} or set(test_labels) != {True, False}:
        raise AssertionError("Both splits must contain answerable true and false")


def print_summary(name, records):
    labels = Counter(record["answerable"] for record in records)
    print(
        f"{name:<6} total={len(records):>4}  "
        f"answerable=true={labels[True]:>4}  "
        f"answerable=false={labels[False]:>3}"
    )

    by_grade = Counter(
        (str(record["grade"]), record["answerable"]) for record in records
    )
    for grade in sorted(
        {key[0] for key in by_grade},
        key=lambda value: int(value) if value.isdigit() else math.inf,
    ):
        print(
            f"  grade {grade:<2}  true={by_grade[(grade, True)]:>3}  "
            f"false={by_grade[(grade, False)]:>2}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create grouped and stratified QA train/test JSONL splits."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    records, input_files = load_records(args.input_dir)
    totals, targets, target_total = allocate_test_targets(records)
    _, groups_by_grade = group_records(records)
    selected_contexts = select_test_groups(groups_by_grade, targets, args.seed)
    train_records, test_records = split_records(
        records,
        selected_contexts,
        args.seed,
    )

    if len(test_records) != target_total:
        raise AssertionError(
            f"Expected {target_total} test records, got {len(test_records)}"
        )

    validate_split(records, train_records, test_records, targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "test.jsonl", test_records)

    print("Input files:", ", ".join(path.name for path in input_files))
    print("Seed:", args.seed)
    print("Source strata:", dict(sorted(totals.items())))
    print_summary("train", train_records)
    print_summary("test", test_records)
    print("Shared train/test contexts: 0")
    print("Wrote:", (args.output_dir / "train.jsonl").resolve())
    print("Wrote:", (args.output_dir / "test.jsonl").resolve())


if __name__ == "__main__":
    main()
