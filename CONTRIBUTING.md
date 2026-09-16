# Contributing

Thank you for contributing.

## Before submitting

- Do not upload patient-identifiable X-rays or health information.
- Read `README.md`, `SECURITY.md`, and `THIRD_PARTY_NOTICES.md`.
- Document the source and license of third-party code, models, and data.

## Development

Target Python version: 3.10.

Run checks:

```bash
python -m compileall -q Bone-pre.py bone_age/bone_age.py models utils export.py tests
python -m pytest -q
```

## High-impact changes

Changes to scoring tables, ROI mapping, model weights, or data provenance require reproducible evidence and documentation updates.

## Pull requests

Keep changes focused. Include motivation, validation steps, and any compatibility impact.
