# VideoChat-TPO Inference (Clean Codebase)

This repo provides a **clean inference-only pipeline** for the CVPR 2025 paper  
**"Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment"**  
using the Hugging Face model [`OpenGVLab/VideoChat-TPO`](https://huggingface.co/OpenGVLab/VideoChat-TPO).

It does **not** depend on the original GitHub repo (which is currently broken), except conceptually.

---

## 1. Setup

```bash
git clone <this-repo-url>
cd videochat_tpo_infer

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -r requirements.txt
