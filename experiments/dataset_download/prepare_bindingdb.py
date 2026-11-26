"""Stream a frozen BindingDB TSV into normalized interaction tables.

The preparation step retains Ki, Kd, IC50, and EC50 as typed observations,
requires one protein chain with a sequence, canonicalizes ligand SMILES, and
records every rejected source measurement with an explicit reason.

Outputs
-------
``ligands.tsv.gz``
    One row per stable ligand ID and canonical SMILES.
``targets.tsv.gz``
    One row per sequence-derived target ID.
``interactions.tsv.gz``
    One row per retained typed affinity observation.
``measurements.tsv.gz``
    One row per source measurement with retained/excluded status.
``exclusions.tsv.gz``
    One row per rejected source measurement and its reason.
``preparation_manifest.json``
    Source hash, schema, counts, timings, and the source-accounting invariant.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import resource
import sqlite3
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import pandas as pd
from rdkit import Chem, RDLogger

from experiments.dataset_download.download_bindingdb import ROOT, sha256_file, validate_archive


RDLogger.DisableLog("rdApp.*")
DEFAULT_OUTPUT = ROOT / "data/processed/bindingdb"
AFFINITY_COLUMNS = {
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
}
DEFAULT_AFFINITY_TYPES = tuple(AFFINITY_COLUMNS)
REQUIRED_COLUMNS = {
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "BindingDB MonomerID",
    "Target Name",
}
SEQUENCE_PREFIX = "BindingDB Target Chain Sequence"
UNIPROT_PREFIXES = (
    "UniProt (SwissProt) Primary ID of Target Chain",
    "UniProt (TrEMBL) Primary ID of Target Chain",
)
RELATION_VALUE = re.compile(
    r"^\s*(?P<relation><=|>=|<|>|~|=)?\s*(?P<value>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*$"
)


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _base_column(column: str) -> str:
    """Undo pandas' suffix for duplicate BindingDB chain columns."""
    return re.sub(r"\.\d+$", "", column)


def matching_columns(columns: Iterable[str], prefix: str) -> list[str]:
    """Match both legacy duplicate headers and numbered 2026 chain headers."""
    return [
        column for column in columns
        if _base_column(column) == prefix
        or re.fullmatch(re.escape(prefix) + r"\s+\d+", column) is not None
    ]


def parse_affinity(raw: object) -> tuple[str, float] | None:
    text = _clean(raw).replace(",", "")
    match = RELATION_VALUE.match(text)
    if not match:
        return None
    value = float(match.group("value"))
    if not math.isfinite(value) or value <= 0:
        return None
    return match.group("relation") or "=", value


def canonical_smiles(raw: object) -> str | None:
    text = _clean(raw)
    molecule = Chem.MolFromSmiles(text) if text else None
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def normalized_sequences(row: pd.Series, sequence_columns: list[str]) -> list[str]:
    sequences = []
    for column in sequence_columns:
        sequence = re.sub(r"[^A-Za-z]", "", _clean(row.get(column))).upper()
        if sequence and sequence not in sequences:
            sequences.append(sequence)
    return sequences


def stable_hash(prefix: str, *values: object, length: int = 24) -> str:
    payload = "\x1f".join(str(v) for v in values).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def peak_rss_mb() -> float:
    """Return process peak resident memory in MiB on macOS or Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_used = value if sys.platform == "darwin" else value * 1024
    return bytes_used / (1024 * 1024)


def open_bindingdb_table(path: Path, member: str | None = None) -> tuple[TextIO, object]:
    """Open a TSV directly or the selected tabular member of a zip archive."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        validate_archive(path)
        archive = zipfile.ZipFile(path)
        candidates = [
            name for name in archive.namelist()
            if name.lower().endswith((".tsv", ".txt")) and not name.endswith("/")
        ]
        if member:
            if member not in candidates:
                archive.close()
                raise ValueError(f"Requested member {member!r} not found in {path}")
            selected = member
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            archive.close()
            raise ValueError(
                "Archive contains multiple tabular files; select one with --member: "
                + ", ".join(candidates)
            )
        binary = archive.open(selected)
        import io
        return io.TextIOWrapper(binary, encoding="utf-8", errors="replace", newline=""), archive
    handle = path.open("r", encoding="utf-8", errors="replace", newline="")
    return handle, handle


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE ligands (
            ligand_id TEXT PRIMARY KEY,
            monomer_id TEXT NOT NULL,
            canonical_smiles TEXT NOT NULL,
            inchikey TEXT
        );
        CREATE TABLE targets (
            target_id TEXT PRIMARY KEY,
            target_name TEXT NOT NULL,
            sequence TEXT NOT NULL,
            uniprot_id TEXT
        );
        CREATE TABLE measurements (
            source_measurement_id TEXT PRIMARY KEY,
            source_row INTEGER NOT NULL UNIQUE,
            reactant_set_id TEXT,
            status TEXT NOT NULL,
            exclusion_reason TEXT,
            observation_count INTEGER NOT NULL
        );
        CREATE TABLE interactions (
            observation_id TEXT PRIMARY KEY,
            source_measurement_id TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            reactant_set_id TEXT,
            ligand_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            affinity_type TEXT NOT NULL,
            affinity_relation TEXT NOT NULL,
            affinity_nm REAL NOT NULL,
            p_affinity REAL NOT NULL,
            duplicate_group_id TEXT NOT NULL
        );
        CREATE INDEX interactions_ligand ON interactions(ligand_id);
        CREATE INDEX interactions_target ON interactions(target_id);
        CREATE INDEX interactions_duplicate ON interactions(duplicate_group_id);
        CREATE TABLE exclusions (
            source_row INTEGER PRIMARY KEY,
            reactant_set_id TEXT,
            reason TEXT NOT NULL
        );
        """
    )
    return connection


def _write_gzip_tsv(connection: sqlite3.Connection, query: str, output: Path) -> int:
    cursor = connection.execute(query)
    columns = [item[0] for item in cursor.description]
    count = 0
    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        while rows := cursor.fetchmany(50_000):
            writer.writerows(rows)
            count += len(rows)
    return count


def prepare_bindingdb(
    source: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    affinity_types: Iterable[str] = DEFAULT_AFFINITY_TYPES,
    member: str | None = None,
    chunk_size: int = 50_000,
    max_rows: int | None = None,
    force: bool = False,
) -> dict:
    """Normalize BindingDB while guaranteeing retained+excluded source accounting."""
    source = Path(source)
    output_dir = Path(output_dir)
    affinity_types = tuple(dict.fromkeys(affinity_types))
    unsupported = set(affinity_types) - set(AFFINITY_COLUMNS)
    if not affinity_types or unsupported:
        raise ValueError(f"Unsupported affinity types: {sorted(unsupported)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "preparation_manifest.json"
    database_path = output_dir / "bindingdb_staging.sqlite"
    managed_outputs = [
        output_dir / "ligands.tsv.gz",
        output_dir / "targets.tsv.gz",
        output_dir / "measurements.tsv.gz",
        output_dir / "interactions.tsv.gz",
        output_dir / "exclusions.tsv.gz",
        manifest_path,
        database_path,
    ]
    existing = [p for p in managed_outputs if p.exists()]
    if existing and not force:
        raise FileExistsError(
            f"BindingDB prepared outputs already exist ({existing[0]}); pass --force to replace them"
        )
    for path in existing:
        path.unlink()

    started = time.perf_counter()
    stage_started = started
    header_handle, header_owner = open_bindingdb_table(source, member)
    try:
        source_columns = list(
            pd.read_csv(header_handle, sep="\t", dtype=str, nrows=0).columns
        )
    finally:
        header_handle.close()
        if header_owner is not header_handle:
            header_owner.close()
    affinity_columns = {kind: AFFINITY_COLUMNS[kind] for kind in affinity_types}
    missing = (REQUIRED_COLUMNS | set(affinity_columns.values())) - set(source_columns)
    sequence_columns = matching_columns(source_columns, SEQUENCE_PREFIX)
    if missing or not sequence_columns:
        details = sorted(missing | ({SEQUENCE_PREFIX} if not sequence_columns else set()))
        raise ValueError(f"Unexpected BindingDB schema; missing {details}")
    uniprot_columns = [
        column
        for prefix in UNIPROT_PREFIXES
        for column in matching_columns(source_columns, prefix)
    ]
    inchikey_column = "Ligand InChI Key" if "Ligand InChI Key" in source_columns else None
    selected_columns = list(REQUIRED_COLUMNS | set(affinity_columns.values())) + sequence_columns + uniprot_columns
    if inchikey_column:
        selected_columns.append(inchikey_column)
    # Keep order stable while removing overlaps.
    selected_columns = list(dict.fromkeys(selected_columns))

    handle, owner = open_bindingdb_table(source, member)
    connection = initialize_database(database_path)
    exclusions = Counter()
    observation_counts = Counter()
    source_rows = 0
    try:
        reader = pd.read_csv(
            handle,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
            keep_default_na=False,
            low_memory=False,
            on_bad_lines="error",
            usecols=selected_columns,
        )
        for chunk in reader:
            if max_rows is not None:
                remaining = max_rows - source_rows
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]
            ligand_rows: list[tuple] = []
            target_rows: list[tuple] = []
            measurement_rows: list[tuple] = []
            interaction_rows: list[tuple] = []
            exclusion_rows: list[tuple] = []
            smiles_cache: dict[str, str | None] = {}
            for _, row in chunk.iterrows():
                source_rows += 1
                reactant_id = _clean(row.get("BindingDB Reactant_set_id"))
                parsed_affinities = {
                    kind: parsed
                    for kind, column in affinity_columns.items()
                    if (parsed := parse_affinity(row.get(column))) is not None
                }
                sequences = normalized_sequences(row, sequence_columns)
                raw_smiles = _clean(row.get("Ligand SMILES"))
                if raw_smiles not in smiles_cache:
                    smiles_cache[raw_smiles] = canonical_smiles(raw_smiles)
                smiles = smiles_cache[raw_smiles]
                source_measurement_id = (
                    f"bdb:{reactant_id}:row:{source_rows}"
                    if reactant_id else f"bindingdb:row:{source_rows}"
                )

                reason = None
                if smiles is None:
                    reason = "invalid_ligand_smiles"
                elif len(sequences) == 0:
                    reason = "missing_target_sequence"
                elif len(sequences) > 1:
                    reason = "multichain_target"
                elif not parsed_affinities:
                    reason = "no_valid_affinity"
                if reason:
                    exclusions[reason] += 1
                    exclusion_rows.append((source_rows, reactant_id, reason))
                    measurement_rows.append((
                        source_measurement_id, source_rows, reactant_id,
                        "excluded", reason, 0,
                    ))
                    continue

                sequence = sequences[0]
                monomer_id = _clean(row.get("BindingDB MonomerID"))
                ligand_id = (
                    f"bdbm:{monomer_id}" if monomer_id
                    else stable_hash("ligand", smiles)
                )
                target_name = _clean(row.get("Target Name"))
                target_id = stable_hash("target", sequence)
                uniprot = next(
                    (_clean(row.get(column)) for column in uniprot_columns if _clean(row.get(column))),
                    "",
                )
                ligand_rows.append((
                    ligand_id, monomer_id, smiles,
                    _clean(row.get(inchikey_column)) if inchikey_column else "",
                ))
                target_rows.append((target_id, target_name, sequence, uniprot))
                measurement_rows.append((
                    source_measurement_id, source_rows, reactant_id,
                    "retained", None, len(parsed_affinities),
                ))
                for affinity_type, (relation, affinity_nm) in parsed_affinities.items():
                    observation_counts[affinity_type] += 1
                    observation_id = f"{source_measurement_id}:{affinity_type}"
                    p_affinity = 9.0 - math.log10(affinity_nm)
                    duplicate_group_id = stable_hash(
                        "duplicate", ligand_id, target_id, affinity_type,
                        relation, f"{affinity_nm:.12g}"
                    )
                    interaction_rows.append((
                        observation_id, source_measurement_id, source_rows,
                        reactant_id, ligand_id, target_id, affinity_type,
                        relation, affinity_nm, p_affinity, duplicate_group_id,
                    ))

            with connection:
                connection.executemany(
                    "INSERT OR IGNORE INTO ligands VALUES (?, ?, ?, ?)", ligand_rows
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", target_rows
                )
                connection.executemany(
                    "INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?)", measurement_rows
                )
                connection.executemany(
                    "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    interaction_rows,
                )
                connection.executemany(
                    "INSERT INTO exclusions VALUES (?, ?, ?)", exclusion_rows
                )
            print(
                f"Processed {source_rows:,} measurements; "
                f"emitted {connection.execute('SELECT COUNT(*) FROM interactions').fetchone()[0]:,} observations",
                flush=True,
            )

        normalization_seconds = time.perf_counter() - stage_started
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("ligands", "targets", "measurements", "interactions", "exclusions")
        }
        retained_measurements = connection.execute(
            "SELECT COUNT(*) FROM measurements WHERE status = 'retained'"
        ).fetchone()[0]
        excluded_measurements = connection.execute(
            "SELECT COUNT(*) FROM measurements WHERE status = 'excluded'"
        ).fetchone()[0]
        if counts["measurements"] != source_rows or retained_measurements + excluded_measurements != source_rows:
            raise RuntimeError(
                "Source-accounting invariant failed: retained plus excluded measurements "
                "does not equal the number of input measurements"
            )
        duplicate_measurements = connection.execute(
            """SELECT COALESCE(SUM(n - 1), 0) FROM (
                   SELECT COUNT(*) AS n FROM interactions GROUP BY duplicate_group_id HAVING n > 1
               )"""
        ).fetchone()[0]
        export_started = time.perf_counter()
        exports = {
            "ligands": _write_gzip_tsv(
                connection, "SELECT * FROM ligands ORDER BY ligand_id", output_dir / "ligands.tsv.gz"
            ),
            "targets": _write_gzip_tsv(
                connection, "SELECT * FROM targets ORDER BY target_id", output_dir / "targets.tsv.gz"
            ),
            "measurements": _write_gzip_tsv(
                connection, "SELECT * FROM measurements ORDER BY source_row", output_dir / "measurements.tsv.gz"
            ),
            "interactions": _write_gzip_tsv(
                connection, "SELECT * FROM interactions ORDER BY source_row, affinity_type", output_dir / "interactions.tsv.gz"
            ),
            "exclusions": _write_gzip_tsv(
                connection, "SELECT * FROM exclusions ORDER BY source_row", output_dir / "exclusions.tsv.gz"
            ),
        }
        if exports != counts:
            raise RuntimeError(f"Export cardinality mismatch: database={counts}, exports={exports}")
        export_seconds = time.perf_counter() - export_started
        hash_started = time.perf_counter()
        source_sha256 = sha256_file(source)
        hashing_seconds = time.perf_counter() - hash_started
        elapsed = time.perf_counter() - started
        manifest = {
            "format_version": 1,
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "source_sha256": source_sha256,
            "archive_member": member,
            "affinity_types": list(affinity_types),
            "affinity_unit": "nM",
            "target_policy": "single unique protein sequence per measurement",
            "ligand_policy": "BindingDB MonomerID with RDKit-canonicalized SMILES",
            "source_measurements": source_rows,
            "retained_source_measurements": retained_measurements,
            "excluded_source_measurements": excluded_measurements,
            "affinity_observations": counts["interactions"],
            "affinity_observations_by_type": {
                kind: observation_counts.get(kind, 0) for kind in affinity_types
            },
            "unique_ligands": counts["ligands"],
            "unique_targets": counts["targets"],
            "duplicate_observations": int(duplicate_measurements),
            "exclusion_reasons": dict(sorted(exclusions.items())),
            "source_accounting_valid": True,
            "chunk_size": chunk_size,
            "max_rows": max_rows,
            "runtime_seconds": {
                "normalization_and_staging": normalization_seconds,
                "table_export": export_seconds,
                "source_hashing": hashing_seconds,
                "total": elapsed,
            },
            "peak_memory_mb": peak_rss_mb(),
            "outputs": {
                name: str((output_dir / f"{name}.tsv.gz").relative_to(ROOT))
                if (output_dir / f"{name}.tsv.gz").is_relative_to(ROOT)
                else str(output_dir / f"{name}.tsv.gz")
                for name in exports
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return manifest
    finally:
        connection.close()
        handle.close()
        if owner is not handle:
            owner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BindingDB TSV or frozen zip archive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--affinity-types", nargs="+", choices=list(AFFINITY_COLUMNS),
        default=list(DEFAULT_AFFINITY_TYPES),
        help="Typed affinity observations to retain (default: all four)",
    )
    parser.add_argument("--member", help="TSV member when the zip contains more than one")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, help="Prepare only the first N rows for a pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_bindingdb(
        args.source,
        output_dir=args.output_dir,
        affinity_types=args.affinity_types,
        member=args.member,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        force=args.force,
    )


if __name__ == "__main__":
    main()
