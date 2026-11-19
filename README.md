# StructurLM — Automated IT Ticket Triage API

**StructurLM** is an end-to-end MLOps and Small Language Model (SLM) deployment project that converts unstructured IT support tickets into clean, schema-consistent JSON classifications. It combines a QLoRA fine-tuned **Qwen2.5-1.5B-Instruct** adapter with a containerized **FastAPI** inference service and a fully automated **GitHub Actions → Google Cloud Run** CI/CD pipeline.

---

## Live API & Interactive Demo

The service is deployed on **Google Cloud Run** (`us-central1`) and scales automatically from zero:

* **Interactive Swagger UI:** [https://structura-lm-api-234157723746.us-central1.run.app/docs](https://structura-lm-api-234157723746.us-central1.run.app/docs)
* **Prediction Endpoint:** `POST https://structura-lm-api-234157723746.us-central1.run.app/predict`

> **Note on Demo Mode (`TEST_MODE=true`):** The live CPU-based Cloud Run deployment currently runs with `TEST_MODE=true` to demonstrate the full production API contract and CI/CD pipeline with low latency and zero GPU hosting cost. The fine-tuned QLoRA adapter weights are included in `model/qwen2.5_ticket_lora/` for full model inference when `TEST_MODE=false`.

### Quick Test with `curl`

```bash
curl -X 'POST' \
  [https://structura-lm-api-234157723746.us-central1.run.app/predict](https://structura-lm-api-234157723746.us-central1.run.app/predict) \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "Outlook Crashing on Startup",
    "message": "Since this morning'\''s Windows update, Microsoft Outlook closes immediately with error 0x80040115 whenever I open it."
  }'
```

### Example JSON Response

```json
{
  "department_queue": "IT Support",
  "urgency": "High",
  "ticket_type": "Incident",
  "primary_tag": "App Crash",
  "_note": "This is a fake test response!"
}
```

---

## System Architecture & CI/CD Pipeline

Every push to `main` triggers an automated deployment pipeline with immutable image tagging via Git commit SHA:

```text
Git Push (main)
   │
   ▼
GitHub Actions (.github/workflows/deploy.yml)
   ├── Authenticate to GCP via least-privilege Service Account (GCP_SA_KEY)
   ├── Build container image tagged with Commit SHA + :latest
   ├── Push image to Google Artifact Registry (us-central1)
   └── Roll out new immutable revision to Google Cloud Run (100% traffic)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model & Fine-Tuning** | Qwen2.5-1.5B-Instruct, QLoRA (4-bit NF4 Quantization), Hugging Face PEFT |
| **API & Serving** | FastAPI, Uvicorn, Pydantic |
| **Containerization** | Docker |
| **CI/CD Automation** | GitHub Actions (`google-github-actions/auth`, `setup-gcloud`) |
| **Cloud Infrastructure** | Google Cloud Run, Google Artifact Registry, GCP IAM |

---

## Project Structure

```text
StructurLM/
├── .github/
│   └── workflows/
│       └── deploy.yml          # Automated build, push, and Cloud Run deployment
├── app/                        # FastAPI application and inference routing
├── model/
│   └── qwen2.5_ticket_lora/    # Fine-tuned QLoRA adapter weights
├── Dockerfile                  # Production container specification (port 8000)
├── .dockerignore               # Excludes local artifacts and temporary CI credentials
├── .gitignore                  # Protects virtual envs, caches, and GCP keys
└── requirements.txt            # Python runtime dependencies
```

---

## Run Locally with Docker

```bash
# 1. Clone the repository
git clone [https://github.com/K26J/StructurLM-The-JSON-Extractor.git](https://github.com/K26J/StructurLM-The-JSON-Extractor.git)
cd StructurLM-The-JSON-Extractor

# 2. Build the container image
docker build -t structura-lm-api .

# 3. Run the API in test mode on port 8000
docker run -p 8000:8000 -e TEST_MODE=true structura-lm-api
```

Once running, open `http://localhost:8000/docs` in your browser to test the `/predict` endpoint interactively.