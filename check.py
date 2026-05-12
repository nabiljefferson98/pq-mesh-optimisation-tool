#!/usr/bin/env python3
"""
Project Quality Check Script.
Runs all code quality tools: formatting, linting, types, security, and tests.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:
    print("Error: 'rich' library not found. Please run: pip install rich")
    sys.exit(1)

console = Console()

# Configuration
SOURCES = ["src", "tests", "scripts"]
SRC_ONLY = ["src"]
FAIL_UNDER_COVERAGE = 70


class Checker:
    def __init__(self, fix: bool = False):
        self.fix = fix
        self.results: Dict[str, Tuple[bool, str]] = {}
        self.failed_files: Dict[str, List[str]] = {}

    def run_command(self, name: str, command: List[str]) -> bool:
        """Runs a command and captures its output."""
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            success = process.returncode == 0
            output = process.stdout + process.stderr

            self.results[name] = (success, output)

            if not success:
                # Heuristics to extract filenames from output
                files = set()
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    # 1. Pattern: "path/to/file.py:line:..." (flake8, mypy, pylint,
                    # vulture, bandit)
                    if ":" in line:
                        parts = line.split(":")
                        path_str = parts[0].strip()
                        if Path(path_str).exists() and Path(path_str).suffix == ".py":
                            files.add(path_str)

                    # 2. Pattern for black: "would reformat path/to/file.py"
                    if "would reformat " in line:
                        path_str = line.replace("would reformat ", "").strip()
                        if Path(path_str).exists():
                            files.add(path_str)

                    # 3. Pattern: ERROR message with "Imports are incorrectly sorted"
                    # (isort)
                    if "Imports are incorrectly sorted" in line:
                        words = line.split()
                        for word in words:
                            if word.endswith(".py") and Path(word).exists():
                                files.add(word)

                    # 4. Pattern: ">> Issue: ... in file: path/to/file.py:..." (bandit)
                    if "in file: " in line:
                        try:
                            path_str = line.split("in file: ")[1].split(":")[0].strip()
                            if Path(path_str).exists():
                                files.add(path_str)
                        except IndexError:
                            pass

                    # 5. Pattern: .py files in lines like:
                    # "src/optimisation/gradients.py      45     15    67%"
                    if any(x in line for x in [".py", ".py "]) and (
                        "%" in line or "FAILED" in line
                    ):
                        words = line.split()
                        for word in words:
                            word = word.strip(":,[]()")
                            if word.endswith(".py") and Path(word).exists():
                                files.add(word)

                if files:
                    self.failed_files[name] = sorted(list(files))

            return success
        except FileNotFoundError:
            self.results[name] = (False, f"Tool '{command[0]}' not found.")
            return False

    def run_all(self, selected_checks: List[str] = None):
        all_checks = {
            "Format (black)": (
                ["black", "--check"] + SOURCES if not self.fix else ["black"] + SOURCES
            ),
            "Imports (isort)": (
                ["isort", "--check", "--profile", "black"] + SOURCES
                if not self.fix
                else ["isort", "--profile", "black"] + SOURCES
            ),
            "Lint (flake8)": ["flake8"] + SOURCES,
            "Lint (pylint)": ["pylint"] + SRC_ONLY,
            "Types (mypy)": ["mypy", "--ignore-missing-imports", "--no-strict-optional"]
            + SRC_ONLY,
            "Security (bandit)": ["bandit", "-r"] + SRC_ONLY + ["-ll", "-q"],
            "Complexity (radon)": ["radon", "cc"] + SRC_ONLY + ["-a", "-nb"],
            "Dead Code (vulture)": ["vulture"] + SRC_ONLY + ["--min-confidence", "80"],
            "Pre-commit": ["pre-commit", "run", "--all-files"],
            "Tests (pytest)": [
                "pytest",
                "tests/",
                f"--cov={SRC_ONLY[0]}",
                "--cov-report=term-missing",
                f"--cov-fail-under={FAIL_UNDER_COVERAGE}",
            ],
        }

        checks_to_run = selected_checks if selected_checks else all_checks.keys()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for name in checks_to_run:
                if name in all_checks:
                    task = progress.add_task(
                        description=f"Running {name}...", total=None
                    )
                    self.run_command(name, all_checks[name])
                    status = (
                        "[green]PASSED[/green]"
                        if self.results[name][0]
                        else "[red]FAILED[/red]"
                    )
                    progress.update(task, description=f"{name}: {status}")
                    time.sleep(0.1)

    def display_results(self):
        console.print("\n")
        table = Table(title="Project Quality Check Summary")
        table.add_column("Check", style="cyan")
        table.add_column("Result", justify="center")
        table.add_column("Notes", style="dim")

        all_passed = True
        for name, (success, output) in self.results.items():
            status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
            if not success:
                all_passed = False

            notes = ""
            if name in self.failed_files:
                num_files = len(self.failed_files[name])
                notes = f"{num_files} files affected"
            elif not success and "not found" in output:
                notes = "Tool missing"

            table.add_row(name, status, notes)

        console.print(table)

        if not all_passed:
            console.print(
                "\n[bold red]Failures detected in the following files:[/bold red]"
            )
            for name, files in self.failed_files.items():
                if files:
                    console.print(
                        Panel(
                            f"[bold]{name}[/bold]\n"
                            + "\n".join(f" - {f}" for f in files),
                            border_style="red",
                        )
                    )

            console.print(
                "\n[yellow]Tip: Run with --verbose to see full output, or"
                " --fix to auto-format.[/yellow]"
            )
        else:
            console.print(
                "\n[bold green]✨ All checks passed! Your code is looking"
                " great. ✨[/bold green]"
            )

    def display_verbose(self):
        for name, (success, output) in self.results.items():
            if not success:
                console.print(
                    Panel(output, title=f"Output: {name}", border_style="red")
                )


def main():
    parser = argparse.ArgumentParser(description="Run all project quality checks.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix formatting issues (runs black and isort).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full output of failed tools."
    )
    parser.add_argument("--only", nargs="+", help="Run only specific checks.")
    args = parser.parse_args()

    checker = Checker(fix=args.fix)

    try:
        checker.run_all(selected_checks=args.only)
        checker.display_results()

        if args.verbose:
            checker.display_verbose()

        # Exit with error if any non-optional check failed
        # (We treat everything as mandatory for now)
        if any(not res[0] for res in checker.results.values()):
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Check cancelled by user.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
