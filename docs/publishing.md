# Publishing `pptrain`

This repository includes a GitHub Actions workflow at `.github/workflows/publish.yml` that builds the package, runs `twine check`, and publishes through PyPI Trusted Publishing.

## One-Time Setup

1. On TestPyPI, add a GitHub Actions Trusted Publisher for the `pptrain` project.
2. Use owner `Axym-Labs`, repository `pptrain`, workflow `.github/workflows/publish.yml`, and environment `testpypi`.
3. Repeat the same setup on PyPI with environment `pypi`.
4. If the project does not exist on PyPI or TestPyPI yet, create a pending publisher instead of an existing-project publisher.

## TestPyPI Release

1. Bump `version` in `pyproject.toml`.
2. Commit and push the release candidate.
3. In GitHub Actions, run the `Publish` workflow manually and choose `testpypi`.
4. Verify the uploaded package with:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pptrain
```

## PyPI Release

1. Confirm the TestPyPI artifact installs and the README renders correctly.
2. Create a GitHub release for the version tag.
3. The `Publish` workflow will build once and publish to the `pypi` environment automatically.
4. Verify the public install with:

```bash
pip install pptrain
```
