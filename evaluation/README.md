# Reproducible Evaluation

This directory provides a reproducible evaluation runner for labeled hand X-ray data.

The repository does **not** bundle or claim a clinical benchmark dataset. Use a dataset only when you have permission to process it and can document its provenance and evaluation protocol.

## Manifest

Create a CSV with:

```csv
image,sex,age_years
/path/to/deidentified-xray-001.jpg,boy,12.5
/path/to/deidentified-xray-002.jpg,girl,10.8
```

Relative image paths are resolved from the manifest directory.

## Run

```bash
python evaluation/evaluate_dataset.py \
  --manifest /path/to/manifest.csv \
  --output-dir evaluation/results \
  --device 0
```

CPU is supported with `--device cpu`, but full evaluation is substantially slower.

## Outputs

- `predictions.csv`: per-sample prediction, absolute error, and inference failures.
- `summary.json`: overall and sex-stratified metrics.

Reported metrics:

- MAE (years)
- RMSE (years)
- Median absolute error (years)
- Mean bias (years)
- Fraction within ±0.5 year
- Fraction within ±1.0 year
- Deterministic bootstrap 95% CI for MAE

## Reporting results

When publishing results, record at minimum:

- dataset name/version and access source
- train/validation/test split or external-test definition
- number of evaluated and failed samples
- age range and population
- image acquisition/source details when available
- model/repository commit
- confidence and IoU thresholds
- hardware and software environment

Do not compare numbers across studies as if they were directly equivalent unless the population, split, preprocessing, and metric definition are comparable.
