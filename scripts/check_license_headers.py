# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

"""Verify licenses on tracked and newly added source files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APACHE_HEADER = (
    "Copyright Institute for Automotive Engineering (ika), RWTH Aachen University",
    "SPDX-License-Identifier: Apache-2.0",
)
FILE_LICENSES = {
    Path("include/triton_cpp/shm_utils.hpp"): (
        "Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.",
        "SPDX-License-Identifier: BSD-3-Clause",
    )
}
COMMENT_PREFIXES = {
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".cuh": "//",
    ".cu": "//",
    ".cxx": "//",
    ".h": "//",
    ".hpp": "//",
    ".py": "#",
}


def source_files(repository: Path) -> list[Path]:
    """Return source files tracked by Git or newly added to the repository."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode().split("\0")
    return sorted(repository / path for path in paths if path and Path(path).suffix in COMMENT_PREFIXES)


def main() -> int:
    """Report source files that do not start with their required license header."""
    repository = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in source_files(repository):
        relative_path = path.relative_to(repository)
        license_lines = FILE_LICENSES.get(relative_path, APACHE_HEADER)
        prefix = COMMENT_PREFIXES[path.suffix]
        expected = [f"{prefix} {line}" for line in license_lines]
        actual = path.read_text(encoding="utf-8").splitlines()[:2]
        if actual != expected:
            failures.append(str(relative_path))

    if failures:
        print("The following source files do not begin with the required license header:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All source files contain their required license header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
