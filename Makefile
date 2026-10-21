PYTHON ?= python

.PHONY: install example test test-all

install:
	$(PYTHON) -m pip install -e '.[test]'

example:
	$(PYTHON) examples/toy_campaign.py

test:
	$(PYTHON) -m pytest -q \
	  tests/test_product_skeleton.py \
	  tests/test_toy_product_flow.py \
	  tests/test_campaign_compiler.py \
	  tests/test_campaign_memory.py \
	  tests/test_canonical_import_isolation.py \
	  tests/test_promotion_program.py \
	  tests/test_public_problem_bundle.py \
	  tests/test_canonical_candidate_boundary.py \
	  tests/test_canonical_session_lifecycle.py

test-all:
	$(PYTHON) -m pytest tests

