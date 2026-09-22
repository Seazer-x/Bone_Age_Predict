# Bone_Age_Predict

[![CI](https://github.com/Seazer-x/Bone_Age_Predict/actions/workflows/python-app.yml/badge.svg)](https://github.com/Seazer-x/Bone_Age_Predict/actions/workflows/python-app.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)

基于 **YOLOv5 + RUS-CHN** 的手部 X 光骨龄研究与演示项目。项目使用一个 YOLOv5 检测模型定位手部骨骼区域，再使用 9 个分类模型对 13 个 RUS-CHN 评分部位进行成熟度分级，最后计算 CHN 总分与估算骨龄。

> [!IMPORTANT]
> **仅用于研究、教学与软件工程演示。** 本项目不是医疗器械，未提供临床验证结论，输出不能替代放射科、儿科或其他合格医疗专业人员的诊断与判断。

![Bone Age Predict screenshot](https://github.com/user-attachments/assets/48e17396-5082-4feb-95c9-23fb7215d09b)

## 项目状态

- 维护状态：维护中
- 主要维护者：[@Seazer-x](https://github.com/Seazer-x)
- Python：建议 **3.10**（代码使用了 Python 3.10+ 语法，且与当前锁定的 PyTorch 1.13 系列兼容）
- 正式 Release：[v1.0.0](https://github.com/Seazer-x/Bone_Age_Predict/releases/tag/v1.0.0)；发布流程见 [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md)
- 模型性能：当前仓库尚未发布可复现的临床性能评估，详见 [Model Card](docs/MODEL_CARD.md)

## 功能

- Streamlit Web 界面上传 `png/jpg/jpeg` 单手正位 X 光图。
- YOLOv5 检测模型定位手部骨骼 ROI。
- 9 个 YOLOv5 分类模型完成 Radius、Ulna、MCP、PIP、MIP、DIP 等成熟度分级。
- 基于 RUS-CHN 计分表计算 13 个评分部位的分值、CHN 总分和骨龄估算。
- 支持调整检测置信度阈值和 IoU 阈值。
- 检测失败时输出缺失部位和检测计数，便于排查模型输入问题。

## 推理流程

```text
手部 X 光图
    ↓
YOLOv5 检测模型
    ↓
21 个候选手骨 ROI
    ↓
筛选 13 个 RUS-CHN 评分 ROI
    ↓
9 个 YOLOv5 分类模型
    ↓
成熟度等级
    ↓
RUS-CHN 分值
    ↓
CHN 总分 → 骨龄估算
```

核心实现见 [`bone_age/bone_age.py`](bone_age/bone_age.py)。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Seazer-x/Bone_Age_Predict.git
cd Bone_Age_Predict
```

### 2. 创建 Python 3.10 虚拟环境

Linux / macOS：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 启动 Streamlit

```bash
streamlit run Bone-pre.py
```

浏览器打开 Streamlit 显示的本地地址即可。

## 使用方式

1. 在左侧上传单手正位 X 光图像。
2. 选择 `boy` 或 `girl`。这是当前 RUS-CHN 评分实现所需的算法输入，不应理解为对性别概念的医学扩展定义。
3. 根据需要调整：
   - 置信度阈值，默认 `0.40`；
   - IoU 阈值，默认 `0.45`。
4. 默认上传后自动推理，也可以关闭自动推理后手动点击“开始推理”。
5. 检测不足时，可先检查图像是否为合适的单手正位 X 光，再谨慎降低置信度阈值。

## 仓库结构

```text
Bone_Age_Predict/
├── Bone-pre.py                 # Streamlit 应用入口
├── bone_age/
│   ├── bone_age.py             # RUS-CHN 推理、评分与骨龄计算逻辑
│   ├── bone_age.pt             # 检测模型权重
│   └── */best.pt               # 各部位分类模型权重
├── models/                     # YOLOv5 模型代码（含上游 GPL-3.0 声明）
├── utils/                      # YOLOv5 工具代码（含上游 GPL-3.0 声明）
├── export.py                   # YOLOv5 导出代码
├── tests/                      # 轻量回归测试
├── docs/MODEL_CARD.md          # 模型用途、限制与风险说明
├── THIRD_PARTY_NOTICES.md      # 第三方来源与许可证说明
├── CONTRIBUTING.md             # 贡献指南
├── SECURITY.md                 # 安全漏洞报告政策
└── LICENSE                     # GNU GPL v3
```

> 模型权重体积较大。GitHub Actions 的 CI 使用 sparse checkout，只拉取代码和测试所需文件，不下载模型权重。

## 数据、模型与代码来源

当前项目 README 最初引用了百度飞桨 AI Studio 项目：

- [骨龄计算综合应用：使用飞桨让医生再腾出10分钟](https://aistudio.baidu.com/projectdetail/1485230)
- 原 README 记录的参考作者：`@吖吖查`

仓库中的 `models/`、`utils/`、`export.py` 等文件包含明确的 **“YOLOv5 by Ultralytics, GPL-3.0 license”** 文件头，因此本仓库代码采用 GPL-3.0 兼容方式发布，并保留原始归属说明。

第三方来源、权重和数据集的适用条款并不一定等同于仓库代码许可证。详细说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 负责任使用与隐私

- 不要把项目输出用于独立临床诊断、治疗决策或自动化医疗决策。
- X 光片可能包含可识别的患者信息。公开提交 Issue、PR、截图或测试样本前，请先完成去标识化。
- 不要在 GitHub Issue / PR 中上传真实患者影像、姓名、ID、检查号、出生日期或其他敏感健康信息。
- 如果将应用部署到第三方云服务，请先确认组织的数据保护、医疗数据和跨境传输要求；敏感场景优先本地运行。
- 当前仓库没有公开宣称对不同设备、机构、人群或成像协议具备临床泛化能力。

更多限制见 [Model Card](docs/MODEL_CARD.md)。

## 开发与测试

本仓库 CI 不下载数百 MB 的模型权重，而是执行可快速复现的代码质量检查：

```bash
python -m compileall -q Bone-pre.py bone_age/bone_age.py models utils export.py tests
python -m pytest -q
```

测试重点覆盖 RUS-CHN 评分数据结构、骨龄多项式计算的回归值以及基础源码可解析性。完整模型推理仍需本地模型权重和适合的测试影像。

## 参与贡献

欢迎修复 Bug、完善文档、补充可复现评估、改善兼容性和增强测试。

提交前请阅读：

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

任何新增第三方代码、模型权重或数据都必须同时说明来源与许可条件。

## 维护与发布

维护职责和决策方式见 [MAINTAINERS.md](MAINTAINERS.md)。变更记录见 [CHANGELOG.md](CHANGELOG.md)。正式发布前的检查清单见 [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md)。

## 引用

仓库提供 [`CITATION.cff`](CITATION.cff)。GitHub 支持在仓库页面直接生成引用信息。

## 许可证

仓库代码按 **GNU General Public License v3.0 only (GPL-3.0-only)** 发布，见 [LICENSE](LICENSE)。

YOLOv5 衍生/引入代码的原始版权和许可证声明必须继续保留。数据集、模型权重以及其他第三方材料可能受各自条款约束，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [百度飞桨 AI Studio 骨龄参考项目](https://aistudio.baidu.com/projectdetail/1485230)
- RUS-CHN 骨龄评分方法相关研究与实践

如果你在研究或教学中使用了本项目，欢迎通过 Issue 或 Pull Request 分享可复现的改进、评估结果和适用边界。
