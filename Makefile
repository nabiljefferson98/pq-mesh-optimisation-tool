# ============================================================
#  pq-mesh-optimisation — Developer Quality Checks
#  Usage:
#    make check        full pipeline (fmt → lint → type → security → complexity → deadcode → pre-commit → test)
#    make fmt          auto-fix formatting (black + isort)
#    make lint         flake8 + pylint
#    make type         mypy type-check
#    make security     bandit security scan
#    make complexity   radon cyclomatic complexity + maintainability index
#    make deadcode     vulture dead-code detection
#    make pre-commit   run all pre-commit hooks against every file
#    make test         pytest with coverage
#    make install-dev  install all dev tools and register git hooks
# ============================================================

SOURCES = src/ tests/ scripts/
SRC     = src/
PYTHON  = python
PIP     = pip

.PHONY: check fmt lint type security complexity deadcode pre-commit test install-dev

# ── Full pipeline ────────────────────────────────────────────
check: fmt lint type security complexity deadcode pre-commit test
	@echo ""
	@echo "============================================"
	@echo "  All checks passed."
	@echo "============================================"

# ── Install dev tools ────────────────────────────────────────
install-dev:
	$(PIP) install mypy bandit pylint radon vulture pre-commit
	pre-commit install

# ── Auto-fix formatting ──────────────────────────────────────
fmt:
	@echo ">>> black (auto-format)"
	black $(SOURCES)
	@echo ">>> isort (sort imports)"
	isort $(SOURCES)

# ── Linting ──────────────────────────────────────────────────
lint:
	@echo ">>> flake8"
	flake8 $(SOURCES)
	@echo ">>> pylint"
	pylint $(SRC) --exit-zero

# ── Type checking ────────────────────────────────────────────
type:
	@echo ">>> mypy"
	mypy --ignore-missing-imports --no-strict-optional $(SRC)

# ── Security scan ────────────────────────────────────────────
security:
	@echo ">>> bandit"
	bandit -r $(SRC) -ll

# ── Complexity ───────────────────────────────────────────────
complexity:
	@echo ">>> radon cyclomatic complexity (C/D/E/F ranked functions)"
	radon cc $(SRC) -a -nb
	@echo ">>> radon maintainability index"
	radon mi $(SRC) -s

# ── Dead code ────────────────────────────────────────────────
deadcode:
	@echo ">>> vulture (dead code, >=80% confidence)"
	vulture $(SRC) --min-confidence 80

# ── Pre-commit hooks ─────────────────────────────────────────
pre-commit:
	@echo ">>> pre-commit (all hooks, all files)"
	pre-commit run --all-files

# ── Tests with coverage ──────────────────────────────────────
test:
	@echo ">>> pytest"
	pytest tests/ --cov=$(SRC) --cov-report=term-missing --cov-fail-under=70
