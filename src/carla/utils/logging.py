"""CSV journals with checkpoint row counts for update-boundary recovery."""

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class CsvJournal:
    """Flush each record; on resume remove rows not committed by the checkpoint.

    A backup preserves any discarded tail. Refuse missing committed records instead of
    silently reporting an incomplete experiment. One training process owns each journal.
    """

    def __init__(self, path: Path, fields: Sequence[str], committed_rows: int = 0) -> None:
        self.path = path
        self.fields = list(fields)
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != self.fields:
                    raise ValueError(f"CSV schema mismatch: {path}")
                rows = list(reader)
        if len(rows) < committed_rows:
            raise ValueError(f"missing committed CSV rows: {path}")
        if any(None in row or None in row.values() for row in rows[:committed_rows]):
            raise ValueError(f"malformed committed CSV rows: {path}")
        if len(rows) > committed_rows:
            backup = path.with_suffix(".csv.recovery.bak")
            backup.write_bytes(path.read_bytes())
        if not path.exists() or len(rows) != committed_rows:
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, self.fields)
                writer.writeheader()
                writer.writerows(rows[:committed_rows])
            temporary.replace(path)
        self.rows = committed_rows

    def write(self, record: Mapping[str, Any]) -> None:
        unknown = set(record) - set(self.fields)
        if unknown:
            raise ValueError(f"unknown CSV fields: {sorted(unknown)}")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, self.fields).writerow(record)
        self.rows += 1
