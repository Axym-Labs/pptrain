# Reference Fixtures

This directory is reserved for generator-parity fixtures exported from the
official paper repositories.

Expected flow:

1. Export a JSON fixture that matches `pptrain.reference_parity`.
2. Commit a small deterministic fixture for the task family under test.
3. Compare `pptrain` task outputs against that fixture in exact-equality tests.
