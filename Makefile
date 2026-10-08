PY ?= python
STATE := $(shell mktemp -d 2>/dev/null || echo /tmp/longrun-demo)

.PHONY: install lint typecheck test check demo clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	coverage run -m unittest discover -s tests -t .
	coverage report

check: lint typecheck test

# Offline demo: start a run, kill -9 it mid-flight, resume it, verify the audit chain.
demo:
	@cp -R examples/buggy-project $(STATE)/ws
	@echo "== starting run (will be killed after ~2.5s)"
	@LONGRUN_STATE_DIR=$(STATE)/state longrun run "Fix the failing test" --run-id demo \
		--workspace $(STATE)/ws --script examples/fix-bug.script.json & PID=$$!; \
		sleep 2.5; kill -9 $$PID; wait $$PID 2>/dev/null; echo "== killed with SIGKILL"
	@echo "== resuming"
	@LONGRUN_STATE_DIR=$(STATE)/state longrun resume demo --script examples/fix-bug.script.json
	@echo "== verifying audit chain"
	@LONGRUN_STATE_DIR=$(STATE)/state longrun verify demo
	@echo "== audit trail"
	@LONGRUN_STATE_DIR=$(STATE)/state longrun log demo

clean:
	rm -rf .coverage htmlcov build dist *.egg-info src/*.egg-info
