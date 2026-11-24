# StructurLM — Automated IT Ticket Triage API

**StructurLM** is an end-to-end MLOps and Small Language Model (SLM) deployment system that converts unstructured IT support tickets into strict, schema-validated JSON classifications. It pairs a **QLoRA v3 fine-tuned `Qwen2.5-1.5B-Instruct`** adapter (tracked in **DagsHub MLflow**) with a containerized **FastAPI** inference service and an automated **GitHub Actions → Google Cloud Run** CI/CD pipeline.

Across **600 unseen test tickets**, the v3 adapter achieves **100.0% valid JSON compliance (600/600)**, **79.99% Macro-F1 on `ticket_type`**, and **74.67% accuracy (68.81% Macro-F1) across 16 `primary_tag` classes**.

---

## Live API & Interactive Demo

The service is deployed on **Google Cloud Run** (`us-central1`) and scales automatically from zero:

- **Interactive Swagger UI:** [https://structura-lm-api-234157723746.us-central1.run.app/docs](https://structura-lm-api-234157723746.us-central1.run.app/docs)
- **Health Check Endpoint:** `GET https://structura-lm-api-234157723746.us-central1.run.app/health`
- **Prediction Endpoint:** `POST https://structura-lm-api-234157723746.us-central1.run.app/predict`

> **Note on Demo Mode (`TEST_MODE=true`):** The live CPU-based Cloud Run deployment runs with `TEST_MODE=true` to demonstrate the full production API contract, Pydantic validation, and CI/CD rollout at zero GPU hosting cost. The fine-tuned **v3 QLoRA adapter weights** are packaged inside `model/qwen2.5_ticket_lora/` for live 4-bit NF4 GPU inference when `TEST_MODE=false`.

### Quick Test with `curl`

```bash
curl -X POST \
  [https://structura-lm-api-234157723746.us-central1.run.app/predict](https://structura-lm-api-234157723746.us-central1.run.app/predict) \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "Outlook Crashing on Startup",
    "message": "Since this morning'\''s Windows update, Microsoft Outlook closes immediately with error 0x80040115 whenever I open it."
  }'
```

### Example JSON Response (`TicketResponse` Schema)

```json
{
  "department_queue": "IT Support",
  "urgency": "High",
  "ticket_type": "Incident",
  "primary_tag": "Network"
}
```

---

## Model Training & Engineering Iteration (v2 → v3)

The adapter is fine-tuned on the [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset using a balanced training split (**3,000 training tickets** at 300 per department across 10 queues, **400 validation tickets**, and **600 unseen test tickets**) with Top-15 `primary_tag` classes plus `"Other"` (16 total tags).

### QLoRA v3 Configuration
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` quantized to **4-bit NF4** (`bitsandbytes`, double quantization, FP16 compute dtype).
- **LoRA Adapters:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05` across all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — ~18.4M trainable parameters (~1.18%).
- **v3 Optimization Fixes:** Controlled error analysis on v2 exposed two classic LLM fine-tuning bottlenecks—**loss dilution** (standard full-sequence loss spent ~90% of gradients on system prompt and email tokens) and **LoRA under-training** (`learning_rate=2e-5` for 1 epoch). Upgrading v3 to TRL `SFTTrainer` with **`completion_only_loss=True`** (strict prompt/completion split), **`learning_rate=2e-4`**, and **2 epochs** dropped validation loss from `0.533` (v2) to **`0.123`** (v3, `95.1%` completion token accuracy) with near-zero overfitting (`0.114` train vs. `0.123` val).
- **Experiment Tracking:** Full training curves, hyperparameters, classification reports, and adapter artifacts are tracked in **DagsHub MLflow** ([View v3 Run `644e96f5`](https://dagshub.com/K26J/StructurLM.mlflow/#/experiments/0/runs/644e96f59da946dcbd295b3a8bb2e760)).

---

## Evaluation Benchmarks

### 1. Final Benchmark — Full 600 Unseen Test Tickets (v3)

| Field | Classes | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **JSON Schema Validity** | Strict 4-Key | **100.00%** | — | `600 / 600` valid parses, zero formatting failures |
| **`ticket_type`** | 4 | **79.00%** | **79.99%** | Strong, balanced multi-class generalization |
| **`primary_tag`** | 16 | **74.67%** | **68.81%** | High extraction accuracy across Top-15 + `Other` |
| **`urgency`** | 3 | **43.33%** | **38.97%** | Above 33.3% baseline; SLA priority often external to email |
| **`department_queue`** | 10 | **32.17%** | **32.52%** | Strong on distinct queues; capped by prior shift & overlap |

### 2. Controlled Ablation — v2 vs. v3 (Same 50 Unseen Tickets)

| Metric | v2 (`lr=2e-5`, Full-Seq, 1 Ep) | v3 (`lr=2e-4`, Comp-Only, 2 Ep) | Delta |
| :--- | :---: | :---: | :---: |
| **Valid JSON Rate** | 100.0% | **100.0%** | Held steady |
| **`ticket_type`** (Acc / Macro-F1) | 74.0% / 64.7% | **78.0% / 82.1%** | **+4.0% / +17.4%** |
| **`primary_tag`** (Acc / Macro-F1) | 62.0% / 67.0% | **74.0% / 75.8%** | **+12.0% / +8.8%** |
| **`department_queue`** (Acc / Macro-F1) | 28.0% / 26.9% | **30.0% / 38.1%** | **+2.0% / +11.1%** |

Upgrading to v3 eliminated hallucinated queue names (such as tag leakage into `department_queue`) and recovered previously collapsed classes (`Customer Service` and `Product Support`).

### Error Analysis & Data-Centric Insights

Inspecting the 600-ticket per-queue classification report confirmed that the remaining ceiling on `department_queue` is a **data-distribution and taxonomy bottleneck** rather than under-training:

- **Distinct business queues classify cleanly:** `Billing and Payments` achieves **0.84 precision / 0.76 F1** ($n=61$) and `Service Outages and Maintenance` achieves **0.70 recall / 0.65 F1** ($n=23$).
- **Prior Probability Shift:** Subsampling the training split to an equal 10% prior across all 10 queues caused over-prediction on rare test classes (`General Inquiry`: 8 true tickets, 0.06 precision; `Sales and Pre-Sales`: 18 true tickets, 0.20 precision).
- **4-Queue Semantic Overlap:** `Technical Support`, `Product Support`, `Customer Service`, and `IT Support` make up **448 of the 600 test tickets (74.7%)** and share heavily overlapping ticket phrasing.
- **Planned v4 Data-Centric Upgrade:** Restore natural class priors during fine-tuning (or apply post-hoc prior calibration) and consolidate overlapping Tier-1 support queues into a cleaner business taxonomy.

---

## System Architecture & CI/CD Pipeline

Every push to `main` triggers an automated build and immutable deployment pipeline:

```text
Colab T4 QLoRA Training ──▶ DagsHub MLflow (Metrics, Reports & Adapter Registry)
         │
         ▼
Git Push (main) with v3 Adapter + Synced System Prompt
         │
         ▼
GitHub Actions (.github/workflows/deploy.yml)
   ├── Authenticate to GCP via least-privilege Service Account (GCP_SA_KEY)
   ├── Build container image tagged with Commit SHA + :latest
   ├── Push image to Google Artifact Registry (us-central1)
   └── Roll out immutable revision to Google Cloud Run (port 8000, 100% traffic)
```

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Model & Fine-Tuning** | `Qwen2.5-1.5B-Instruct`, QLoRA (4-bit NF4), Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes`, `accelerate` |
| **Experiment Tracking & Eval** | MLflow, DagsHub, Scikit-Learn, Google Colab (T4 GPU) |
| **API & Serving** | FastAPI, Uvicorn, Pydantic (`TicketRequest` / `TicketResponse`) |
| **Containerization** | Docker (`python:3.10-slim`) |
| **CI/CD Automation** | GitHub Actions (`actions/checkout@v4`, `google-github-actions/auth@v3`, `setup-gcloud@v3`) |
| **Cloud Infrastructure** | Google Cloud Run, Google Artifact Registry, GCP IAM |

---

## Project Structure

```text
StructurLM/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated build, push, and Cloud Run deployment
├── app/
│   └── main.py                 # FastAPI app, v3 prompt sync, /health & /predict routes
├── model/
│   └── qwen2.5_ticket_lora/    # Fine-tuned v3 QLoRA adapter & tokenizer artifacts
├── Dockerfile                  # Production container specification (port 8000)
├── .dockerignore               # Excludes venv, caches, git metadata & CI credentials
├── .gitignore                  # Protects virtual envs, caches, and GCP keys
├── requirements.txt            # Python runtime dependencies
└── README.md                   # Project documentation & v3 evaluation benchmarks
```

---

## Run Locally

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone [https://github.com/K26J/StructurLM-The-JSON-Extractor.git](https://github.com/K26J/StructurLM-The-JSON-Extractor.git)
cd StructurLM-The-JSON-Extractor

# Build the container image
docker build -t structura-lm-api .

# Run in lightweight CPU test mode (port 8000)
docker run -p 8000:8000 -e TEST_MODE=true structura-lm-api

# Or run live 4-bit QLoRA inference on an NVIDIA GPU host
docker run --gpus all -p 8000:8000 -e TEST_MODE=false structura-lm-api
```

### Option 2: Local Python Virtual Environment

```bash
TEST_MODE=true uvicorn app.main:app --reload
```

Once running, open `http://localhost:8000/docs` in your browser to test `/health` and `/predict` interactively.