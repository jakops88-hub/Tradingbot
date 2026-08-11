"""Dataset metadata for normalized historical market data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetMetadata:
    symbol: str
    quote_currency: str
    source: str
    interval: str
    start_date: str
    end_date: str

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if not self.quote_currency:
            raise ValueError("quote_currency is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.interval:
            raise ValueError("interval is required")
        _parse_date(self.start_date, "start_date")
        _parse_date(self.end_date, "end_date")


def metadata_path_for(csv_path: str | Path) -> Path:
    source = Path(csv_path)
    return source.with_suffix(source.suffix + ".metadata.json")


def save_dataset_metadata(csv_path: str | Path, metadata: DatasetMetadata) -> Path:
    path = metadata_path_for(csv_path)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_dataset_metadata(csv_path: str | Path) -> DatasetMetadata | None:
    path = metadata_path_for(csv_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return DatasetMetadata(**data)


def require_matching_currency(
    csv_path: str | Path,
    portfolio_currency: str,
) -> DatasetMetadata | None:
    metadata = load_dataset_metadata(csv_path)
    if metadata is None:
        return None
    if metadata.quote_currency.upper() != portfolio_currency.upper():
        raise ValueError(
            "Portfolio currency and instrument quote currency differ: "
            f"{portfolio_currency.upper()} vs {metadata.quote_currency.upper()}. "
            "FX conversion is not implemented."
        )
    return metadata


def _parse_date(value: str, field_name: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO formatted") from exc
