"""Dump BigQuery dataset schemas as YAML files compatible with the `sql` skill.

The YAML layout matches the existing files under
`.claude/skills/sql/gbq_schema/`: top-level `table: <name>` plus a `fields:`
list, where each field is a mapping of `name`, `type`, optional `mode` (only
emitted when `REPEATED`), `description`, and a recursive `fields:` list for
`RECORD` types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField
from google.cloud.bigquery.table import TableListItem


def schema_field_to_dict(field: SchemaField) -> dict[str, Any]:
    out: dict[str, Any] = {"name": field.name, "type": field.field_type}
    if field.mode == "REPEATED":
        out["mode"] = "REPEATED"
    if field.description:
        out["description"] = field.description
    if field.field_type == "RECORD":
        sub_fields: list[SchemaField] = list(field.fields or ())
        out["fields"] = [schema_field_to_dict(sub) for sub in sub_fields]
    return out


def table_to_dict(table: bigquery.Table) -> dict[str, Any]:
    schema: list[SchemaField] = list(table.schema)
    return {
        "table": table.table_id,
        "fields": [schema_field_to_dict(field) for field in schema],
    }


def dump_dataset_schema(
    client: bigquery.Client,
    dataset_ref: str,
    output_dir: Path,
) -> list[Path]:
    """Write one ``<table>.yaml`` file per table in *dataset_ref* under *output_dir*.

    *dataset_ref* is a fully qualified ``project.dataset`` string. Returns the
    list of files written, in iteration order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    table_refs: list[TableListItem] = list(client.list_tables(dataset_ref))
    for table_ref in table_refs:
        table = client.get_table(table_ref)
        payload = table_to_dict(table)
        path = output_dir / f"{table.table_id}.yaml"
        with path.open("w") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True, width=100)
        written.append(path)
    return written
