"""Migrate pre-0.2 QAssemble HDF5 class groups in place."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

import h5py


# Historical names are intentionally confined to this migration module.
LEGACY_CLASS_GROUPS = {
    "NIHamiltonian": "H0",
    "Hamiltonian": "H",
    "SigmaHartree": "SigH",
    "SigmaFock": "SigF",
    "GreenBare": "G0",
    "GreenInt": "G",
    "SigmaGWC": "SigGWC",
    "PolLat": "P",
    "WLat": "W",
    "VBare": "V",
    "ZFactor": "Z",
    "SigmaStc": "SigStc",
}


class MigrationError(RuntimeError):
    """Raised when an HDF5 file cannot be migrated safely."""


def _migration_targets(path: Path) -> list[tuple[str, str, str]]:
    """Return ``(parent_path, old_name, new_name)`` migrations after validation."""

    targets: list[tuple[str, str, str]] = []
    try:
        with h5py.File(path, "r") as h5file:
            for root_name, root_object in h5file.items():
                if root_name == "input" or not isinstance(root_object, h5py.Group):
                    continue
                for old_name, new_name in LEGACY_CLASS_GROUPS.items():
                    has_old = old_name in root_object
                    has_new = new_name in root_object
                    if has_old and has_new:
                        raise MigrationError(
                            f"Name collision in {root_object.name}: "
                            f"both {old_name!r} and {new_name!r} exist."
                        )
                    if has_old:
                        if not isinstance(root_object[old_name], h5py.Group):
                            raise MigrationError(
                                f"Expected {root_object.name}/{old_name} to be an HDF5 group."
                            )
                        targets.append((root_object.name, old_name, new_name))
    except OSError as exc:
        raise MigrationError(f"Cannot open HDF5 file {path}: {exc}") from exc
    return targets


def _apply_targets(path: Path, targets: list[tuple[str, str, str]]) -> None:
    """Apply the selected HDF5 migration targets in place."""
    with h5py.File(path, "r+") as h5file:
        for parent_path, old_name, new_name in targets:
            parent = h5file[parent_path]
            object_address = h5py.h5o.get_info(parent[old_name].id).addr
            parent.move(old_name, new_name)
            migrated_address = h5py.h5o.get_info(parent[new_name].id).addr
            if migrated_address != object_address:
                raise MigrationError(
                    f"HDF5 object identity changed for {parent_path}/{old_name}."
                )
        h5file.flush()


def _verify_targets(path: Path, targets: list[tuple[str, str, str]]) -> None:
    """Verify that migrated HDF5 targets exist and old groups were removed."""
    with h5py.File(path, "r") as h5file:
        for parent_path, old_name, new_name in targets:
            parent = h5file[parent_path]
            if old_name in parent or new_name not in parent:
                raise MigrationError(
                    f"Verification failed for {parent_path}/{old_name} -> {new_name}."
                )


def migrate_file(
    source: str | os.PathLike[str],
    *,
    backup: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> list[tuple[str, str, str]]:
    """Migrate a QAssemble HDF5 file and return the renamed groups."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise MigrationError(f"HDF5 file does not exist: {source_path}")

    targets = _migration_targets(source_path)
    if dry_run or not targets:
        return targets

    backup_path = (
        Path(backup).expanduser().resolve()
        if backup is not None
        else Path(f"{source_path}.pre-class-rename.bak")
    )
    if backup_path == source_path:
        raise MigrationError("Backup path must differ from the source file.")
    if backup_path.exists():
        raise MigrationError(f"Backup already exists: {backup_path}")
    if not backup_path.parent.is_dir():
        raise MigrationError(f"Backup directory does not exist: {backup_path.parent}")

    temporary_path: Path | None = None
    try:
        shutil.copy2(source_path, backup_path)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{source_path.name}.", suffix=".tmp", dir=source_path.parent
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        shutil.copy2(source_path, temporary_path)
        _apply_targets(temporary_path, targets)
        _verify_targets(temporary_path, targets)
        os.replace(temporary_path, source_path)
        temporary_path = None
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if isinstance(exc, MigrationError):
            raise
        raise MigrationError(f"Migration failed; original file was not replaced: {exc}") from exc

    return targets


def _parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the migration tool."""
    parser = argparse.ArgumentParser(
        description="Migrate QAssemble HDF5 class groups to the manuscript names."
    )
    parser.add_argument("file", help="HDF5 file to migrate in place")
    parser.add_argument("--backup", help="Backup path (default: FILE.pre-class-rename.bak)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print changes without writing files"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        targets = migrate_file(args.file, backup=args.backup, dry_run=args.dry_run)
    except MigrationError as exc:
        parser.exit(1, f"error: {exc}\n")

    if not targets:
        print("No legacy QAssemble class groups found.")
        return 0
    action = "Would rename" if args.dry_run else "Renamed"
    for parent_path, old_name, new_name in targets:
        print(f"{action} {parent_path}/{old_name} -> {new_name}")
    if not args.dry_run:
        backup_path = args.backup or f"{Path(args.file).expanduser().resolve()}.pre-class-rename.bak"
        print(f"Backup: {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
