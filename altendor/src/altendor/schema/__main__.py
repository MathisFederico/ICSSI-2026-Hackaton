"""CLI: ``python -m altendor.schema <project.dataset> <output_dir>``.

Loads `.env`, builds a BigQuery client, and dumps one YAML file per table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery

from altendor.schema.dump import dump_dataset_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m altendor.schema",
        description="Dump a BigQuery dataset's schema as YAML files (sql skill format).",
    )
    parser.add_argument(
        "dataset_ref",
        help="Fully qualified `project.dataset` (e.g. altmetric-endorsements.altmetric_on_gbq).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write `<table>.yaml` files into. Created if missing.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Billing project for jobs. Defaults to the project in dataset_ref.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env (default: ./.env).",
    )
    args = parser.parse_args(argv)

    if args.env.is_file():
        load_dotenv(args.env)

    if "." not in args.dataset_ref:
        parser.error("dataset_ref must be `project.dataset`")

    billing_project: str = args.project or args.dataset_ref.split(".", 1)[0]
    client = bigquery.Client(project=billing_project)

    written = dump_dataset_schema(client, args.dataset_ref, args.output_dir)
    for path in written:
        print(path)
    if not written:
        print(f"No tables found in {args.dataset_ref}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
