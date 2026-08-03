from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from sqlalchemy.engine import URL, make_url


MANAGED_BACKENDS = {"mysql", "mariadb"}
SAFE_RESTORE_NAME_HINTS = ("restore", "staging", "test", "sandbox", "rehearsal")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_database_url(value: str | None, label: str) -> URL:
    if not value or not value.strip():
        raise ValueError(f"{label} database URL is required.")
    url = make_url(value)
    backend = url.get_backend_name()
    if backend not in MANAGED_BACKENDS:
        raise ValueError(f"{label} must use MySQL or MariaDB; got {url.drivername}.")
    if not url.database:
        raise ValueError(f"{label} must include a database name.")
    return url


def is_local_host(host: str | None) -> bool:
    return (host or "").lower() in {"", "localhost", "127.0.0.1", "::1"} or str(host or "").lower().endswith(".local")


def redacted_url(url: URL) -> str:
    return url.render_as_string(hide_password=True)


def summarize_url(url: URL) -> dict[str, Any]:
    return {
        "driver": url.drivername,
        "backend": url.get_backend_name(),
        "host": url.host,
        "port": url.port,
        "database": url.database,
        "usernamePresent": bool(url.username),
        "passwordPresent": bool(url.password),
        "redactedUrl": redacted_url(url),
    }


def database_identity(url: URL) -> tuple[str, str, int | None, str]:
    return (url.get_backend_name(), (url.host or "").lower(), url.port, (url.database or "").lower())


def restore_target_name_is_safe(url: URL) -> bool:
    name = (url.database or "").lower()
    return any(hint in name for hint in SAFE_RESTORE_NAME_HINTS)


def mysql_args(binary: str, url: URL) -> list[str]:
    args = [binary]
    if url.host:
        args.extend(["--host", url.host])
    if url.port:
        args.extend(["--port", str(url.port)])
    if url.username:
        args.extend(["--user", url.username])
    args.append(url.database or "")
    return args


def dump_args(url: URL) -> list[str]:
    return [
        "mysqldump",
        "--single-transaction",
        "--routines",
        "--triggers",
        "--events",
        *mysql_args("", url)[1:],
    ]


def env_with_password(url: URL) -> dict[str, str]:
    env = os.environ.copy()
    if url.password:
        env["MYSQL_PWD"] = url.password
    return env


def command_step(name: str, command: list[str], tool_path: str | None, execute: bool) -> dict[str, Any]:
    return {
        "name": name,
        "toolFound": bool(tool_path),
        "toolPath": tool_path,
        "command": command,
        "execute": execute,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    source = parse_database_url(args.source_url or os.getenv("DATABASE_URL"), "source")
    restore = None
    if args.mode in {"restore", "roundtrip"}:
        restore = parse_database_url(args.restore_url or os.getenv("RESTORE_DATABASE_URL"), "restore")

    errors: list[str] = []
    warnings: list[str] = []

    if is_local_host(source.host) and not args.allow_local_source:
        errors.append("Source database host looks local; use --allow-local-source only for local rehearsals.")
    if restore:
        if is_local_host(restore.host) and not args.allow_local_restore:
            errors.append("Restore database host looks local; use --allow-local-restore only for local rehearsals.")
        if database_identity(source) == database_identity(restore):
            errors.append("Restore database must not be the same host/port/database as the source.")
        if not restore_target_name_is_safe(restore) and not args.allow_unsafe_restore_target:
            errors.append(
                "Restore database name should include restore, staging, test, sandbox, or rehearsal "
                "unless --allow-unsafe-restore-target is set."
            )

    backup_path = Path(args.backup_path or f"backups/horse-predictor-{utc_stamp()}.sql")
    tools = {"mysqldump": shutil.which("mysqldump"), "mysql": shutil.which("mysql")}
    steps: list[dict[str, Any]] = []

    if args.mode in {"backup", "roundtrip"}:
        steps.append(command_step("backup", dump_args(source), tools["mysqldump"], args.execute))
        if args.execute and not tools["mysqldump"]:
            errors.append("mysqldump was not found on PATH.")
    if args.mode in {"restore", "roundtrip"} and restore:
        steps.append(command_step("restore", mysql_args("mysql", restore), tools["mysql"], args.execute))
        if args.execute and not tools["mysql"]:
            errors.append("mysql client was not found on PATH.")
        if args.execute and not backup_path.exists():
            errors.append(f"Backup file does not exist: {backup_path}")

    if args.mode == "plan" and restore is None:
        warnings.append("No restore URL was supplied; the plan covers source validation only.")

    return {
        "status": "failed" if errors else ("ready" if args.execute else "planned"),
        "mode": args.mode,
        "execute": args.execute,
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backupPath": str(backup_path),
        "source": summarize_url(source),
        "restore": summarize_url(restore) if restore else None,
        "tools": tools,
        "steps": steps,
        "warnings": warnings,
        "errors": errors,
    }


def run_step(step: dict[str, Any], url: URL, backup_path: Path) -> None:
    if step["name"] == "backup":
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open("wb") as output:
            subprocess.run(step["command"], stdout=output, stderr=subprocess.PIPE, env=env_with_password(url), check=True)
    elif step["name"] == "restore":
        with backup_path.open("rb") as input_file:
            subprocess.run(step["command"], stdin=input_file, stderr=subprocess.PIPE, env=env_with_password(url), check=True)


def execute_plan(plan: dict[str, Any], source_url: URL, restore_url: URL | None) -> dict[str, Any]:
    backup_path = Path(plan["backupPath"])
    for step in plan["steps"]:
        if step["name"] == "backup":
            run_step(step, source_url, backup_path)
        elif step["name"] == "restore":
            if restore_url is None:
                raise RuntimeError("Restore step requires restore URL.")
            run_step(step, restore_url, backup_path)
    plan["status"] = "passed"
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run a managed MySQL backup/restore rehearsal.")
    parser.add_argument("--mode", choices=["plan", "backup", "restore", "roundtrip"], default="plan")
    parser.add_argument("--source-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--restore-url", default=os.getenv("RESTORE_DATABASE_URL", ""))
    parser.add_argument("--backup-path")
    parser.add_argument("--execute", action="store_true", help="Actually run mysqldump/mysql. Default is dry-run planning.")
    parser.add_argument("--allow-local-source", action="store_true")
    parser.add_argument("--allow-local-restore", action="store_true")
    parser.add_argument("--allow-unsafe-restore-target", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(args)
        if args.execute and not plan["errors"]:
            source = parse_database_url(args.source_url or os.getenv("DATABASE_URL"), "source")
            restore = parse_database_url(args.restore_url or os.getenv("RESTORE_DATABASE_URL"), "restore") if args.mode in {"restore", "roundtrip"} else None
            plan = execute_plan(plan, source, restore)
    except (subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        payload = {"status": "failed", "errors": [str(exc)]}
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        print(rendered)
        return 1

    rendered = json.dumps(plan, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if plan["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
