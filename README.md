# 🦴 Bone_Age_Predict

[![CI](https://github.com/Seazer-x/Bone_Age_Predict/actions/workflows/python-app.yml/badge.svg)](https://github.com/Seazer-x/Bone_Age_Predict/actions/workflows/python-app.yml)
[![Release](https://img.shields.io/github/v/release/Seazer-x/Bone_Age_Predict)](https://github.com/Seazer-x/Bone_Age_Predict/releases/tag/v1.0.0)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)

Open-source **YOLOv5 + RUS-CHN bone age estimation** from hand X-ray images, with a Streamlit demo for research, teaching, and reproducible experimentation.

[中文说明](README.zh-CN.md) · [v1.0.0](https://github.com/Seazer-x/Bone_Age_Predict/releases/tag/v1.0.0) · [Model Card](docs/MODEL_CARD.md) · [Contributing](CONTRIBUTING.md)

> [!IMPORTANT]
> **Research and educational use only.** This project is not a medical device and has not been clinically validated for diagnosis or treatment decisions.

![Bone Age Predict screenshot](https://github.com/user-attachments/assets/48e17396-5082-4feb-95c9-23fb7215d09b)

## What it does

- Detects hand-bone regions with a YOLOv5 detector.
- Selects the 13 RUS-CHN scoring regions from the detected ROIs.
- Uses 9 YOLOv5 classification models to estimate maturation stages.
- Converts stage scores into a CHN total score and estimated bone age.
- Provides a Streamlit UI with adjustable confidence and IoU thresholds.
- Reports missing detections and counts to make failures easier to inspect.

## Pipeline

```text
Hand X-ray
  ↓
YOLOv5 detector
  ↓
Candidate bone ROIs
  ↓
13 RUS-CHN scoring ROIs
  ↓
9 YOLOv5 classifiers
  ↓
Maturation stages
  ↓
RUS-CHN score
  ↓
Estimated bone age
```

Core inference logic: [`bone_age/bone_age.py`](bone_age/bone_age.py)

## Quick start

### 1. Clone

```bash
git clone https://github.com/Seazer-x/Bone_Age_Predict.git
cd Bone_Age_Predict
```

### 2. Create a Python 3.10 environment

Linux / macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the demo

```bash
streamlit run Bone-pre.py
```

Upload a frontal single-hand X-ray image, select the scoring sex input used by the current RUS-CHN implementation, then run inference.

## Hugging Face Space

The hosted demo uses **Gradio + Hugging Face ZeroGPU**. Space-specific files live in [`hf-space/`](hf-space/), while the core inference code remains shared with this repository.

The deployment workflow at [`.github/workflows/deploy-huggingface-space.yml`](.github/workflows/deploy-huggingface-space.yml) creates or updates the ZeroGPU Space using the `HF_TOKEN` GitHub Actions secret. Model weights are downloaded from the tagged `v1.0.0` release at Space startup instead of being duplicated in the Space repository.

## Repository layout

```text
Bone_Age_Predict/
├── Bone-pre.py
├── bone_age/
│   ├── bone_age.py
│   ├── bone_age.pt
│   └── */best.pt
├── models/
├── utils/
├── tests/
├── hf-space/
├── docs/MODEL_CARD.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## Evaluation and limitations

The repository currently does **not** publish a reproducible clinical validation benchmark across hospitals, scanners, age groups, or acquisition protocols. Do not interpret the demo output as a validated clinical measurement.

See [Model Card](docs/MODEL_CARD.md) for intended use, known limitations, and risk notes.

## Data, models, and attribution

The original project documentation referenced:

- [Baidu AI Studio bone-age project](https://aistudio.baidu.com/projectdetail/1485230)
- Ultralytics YOLOv5

Files under `models/`, `utils/`, and `export.py` retain upstream YOLOv5 GPL-3.0 notices. Dataset, model-weight, and third-party terms may differ from the repository code license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Responsible use

- Do not use outputs as an independent clinical diagnosis or treatment decision.
- De-identify X-rays before sharing screenshots, issues, PRs, or examples.
- Do not upload patient names, IDs, dates of birth, accession numbers, or other sensitive health data to public GitHub discussions.
- Prefer local deployment for sensitive data unless your organization has approved the target hosting environment.

## Development

Fast checks:

```bash
python -m compileall -q Bone-pre.py bone_age/bone_age.py models utils export.py tests
python -m pytest -q
```

The CI intentionally avoids downloading model weights and focuses on source validation and lightweight regression tests.

## Contributing

Bug fixes, compatibility improvements, documentation, reproducible evaluations, and tests are welcome.

Please read:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

Repository code is distributed under **GNU General Public License v3.0 only (GPL-3.0-only)**. See [LICENSE](LICENSE).

If this project is useful to your research or teaching, consider starring the repository and sharing reproducible improvements.
