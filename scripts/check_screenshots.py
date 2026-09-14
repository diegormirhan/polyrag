"""Fails when the README's screenshots have gone stale or dangling.

    uv run python scripts/check_screenshots.py

Regenerating them in CI is not possible: scripts/screenshots.py needs three
llama-servers, Qdrant, the backend and a dev server, which means ~4.5 GB of model
weights running on a GPU. A GitHub runner has none of that.

So this checks the two things that can be checked without running anything, and
it is honest about the difference: it does not verify that a screenshot is
*correct*, only that it is not obviously out of date.

Both failures happened in this project. The Corpus view kept showing a chunk
distribution that a later re-ingestion had changed, and a shot of the Telemetry
view was generated and versioned for weeks without ever being referenced.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
README = ROOT / "README.md"
REFERENCE = re.compile(r"docs/screenshots/([\w.-]+\.png)")

# What the pictures actually depict. The interface itself, the corpus they show,
# and the config that decides where that corpus lands and what the panels read.
# Deliberately not all of backend/: a change to the SQL guard does not alter a
# single pixel, and a check that cries wolf gets disabled.
DEPICTED_BY = ["frontend/src", "config.yaml", "demo"]


def last_commit_time(*paths: str) -> int:
    """Commit timestamp of the newest commit touching any of `paths`."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout.strip()
    # Empty when the path has no history at all, which on a shallow clone also
    # means "CI forgot fetch-depth: 0" -- worth saying out loud rather than
    # silently passing.
    if not out:
        raise SystemExit(
            f"git log returned nothing for {paths}. On CI this usually means the "
            "checkout was shallow; the job needs fetch-depth: 0."
        )
    return int(out)


def main() -> None:
    referenced = set(REFERENCE.findall(README.read_text(encoding="utf-8")))
    present = {path.name for path in SHOTS.glob("*.png")}

    problems: list[str] = []
    if missing := sorted(referenced - present):
        problems.append(f"referenced by the README but not in {SHOTS.relative_to(ROOT)}: {missing}")
    if orphans := sorted(present - referenced):
        problems.append(f"committed but referenced nowhere: {orphans}")

    depicted = last_commit_time(*DEPICTED_BY)
    captured = last_commit_time("docs/screenshots")
    if depicted > captured:
        changed = subprocess.run(
            ["git", "log", "-1", "--format=%h %s", "--", *DEPICTED_BY],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        problems.append(
            "the screenshots are older than what they depict.\n"
            f"      last change to {', '.join(DEPICTED_BY)}: {changed}\n"
            "      run: uv run python scripts/screenshots.py  (needs the stack up)"
        )

    if problems:
        print("screenshots:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print(f"screenshots: {len(present)} files, all referenced, none older than what they depict")


if __name__ == "__main__":
    main()
