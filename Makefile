.PHONY: check lint test

check: lint test

lint:
	ruff check .
	ruff format --check .

# pytest exits 5 when no tests are collected; that is green until the first test lands.
test:
	@python -m pytest; rc=$$?; if [ $$rc -ne 0 ] && [ $$rc -ne 5 ]; then exit $$rc; fi
