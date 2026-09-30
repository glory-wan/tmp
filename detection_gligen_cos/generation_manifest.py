"""Append-only generation records shared by workers and the retrain runner."""

import json
import os
from pathlib import Path


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_manifest(output: Path, repair: bool = False) -> list[dict]:
    journal = output / "manifest.jsonl"
    if not journal.exists():
        path = output / "manifest.json"
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(records, list):
            raise ValueError(f"Generation manifest must be a list: {path}")
        return records
    records = []
    with journal.open("r+b" if repair else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # Only newline-terminated records are committed. Discard a torn tail.
                if repair:
                    stream.truncate(offset)
                break
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Generation journal record must be an object: {journal}")
            records.append(record)
    return records


def initialize_journal(output: Path, records: list[dict], resume: bool) -> None:
    journal = output / "manifest.jsonl"
    if resume and journal.exists():
        return
    temporary = journal.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, journal)


def append_record(output: Path, record: dict) -> None:
    with (output / "manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def export_manifest(output: Path, records: list[dict]) -> None:
    write_json_atomic(output / "manifest.json", records)
    write_json_atomic(output / "selected_images.json", [
        str((output / item["result_file"]).resolve())
        for item in records if item.get("result_file")
    ])


def recover_manifest(output: Path) -> None:
    """Rebuild snapshots once before runner resume decisions, never during polling."""
    if (output / "manifest.jsonl").exists():
        export_manifest(output, load_manifest(output))


def progress_count(output: Path) -> int:
    path = output / "progress.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["generated_samples"])
    # Compatibility with workers started before journal support.
    manifest = output / "manifest.json"
    return len(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.exists() else 0
