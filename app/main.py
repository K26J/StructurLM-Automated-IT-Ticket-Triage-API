import os
import json
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

# Allowed labels matching the exact sorted lists used during training
ALLOWED_QUEUES = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]

ALLOWED_PRIORITIES = ["high", "low", "medium"]

ALLOWED_TYPES = ["Change", "Incident", "Problem", "Request"]

ALLOWED_TAGS = [
    "Billing",
    "Bug",
    "Crash",
    "Customer",
    "Disruption",
    "Documentation",
    "Feature",
    "Feedback",
    "Login",
    "Network",
    "Other",
    "Outage",
    "Performance",
    "Product",
    "Security",
    "Technical",
]

# System prompt kept identical to Colab so training and serving stay in sync
SYSTEM_PROMPT = f"""You are an IT ticket triage assistant. Extract structured JSON from the support ticket.
Reply ONLY with valid JSON using these exact keys and allowed values:
- department_queue: {ALLOWED_QUEUES}
- urgency: {ALLOWED_PRIORITIES}
- ticket_type: {ALLOWED_TYPES}
- primary_tag: {ALLOWED_TAGS}"""

# Shared settings and in-memory storage for the loaded model
ml_models = {}
IS_LOCAL_TEST = os.getenv("TEST_MODE", "false").lower() == "true"
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_FOLDER = os.getenv("ADAPTER_PATH", "./model/qwen2.5_ticket_lora")


# Request and response structure expected by the API
class TicketRequest(BaseModel):
    subject: str = ""
    message: str


class TicketResponse(BaseModel):
    department_queue: str
    urgency: str
    ticket_type: str
    primary_tag: str


# Load the base model and LoRA adapter once when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    if IS_LOCAL_TEST:
        print("Running in TEST_MODE: skipping model load to save local RAM.")
        yield
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected and TEST_MODE is off. "
            "Run locally with: TEST_MODE=true uvicorn app.main:app --reload"
        )

    print("GPU detected. Loading compressed base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Loading tokenizer and attaching LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_FOLDER)
    model = PeftModel.from_pretrained(base_model, ADAPTER_FOLDER)
    model.config.use_cache = True
    model.eval()

    ml_models["pipeline"] = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("Model is ready to serve predictions.")

    yield

    ml_models.clear()


# Create the FastAPI app and attach the startup lifecycle
app = FastAPI(title="StructurLM Ticket Triage API", lifespan=lifespan)


# Simple health check route to confirm the server is alive
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "test_mode": IS_LOCAL_TEST,
        "model_loaded": "pipeline" in ml_models,
    }


# Main prediction route that turns a raw support ticket into structured JSON
@app.post("/predict", response_model=TicketResponse)
def predict_ticket(request: TicketRequest):
    if IS_LOCAL_TEST:
        return TicketResponse(
            department_queue="IT Support",
            urgency="High",
            ticket_type="Incident",
            primary_tag="Network",
        )

    subject = request.subject.strip() if request.subject.strip() else "NO Subject"
    user_text = f"Subject: {subject}\nMessage: {request.message}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    try:
        pipe = ml_models["pipeline"]
        pad_id = pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id

        output = pipe(
            messages,
            max_new_tokens=100,
            do_sample=False,
            return_full_text=False,
            use_cache=True,
            pad_token_id=pad_id,
        )

        raw_reply = output[0]["generated_text"].strip()
        parsed_json = json.loads(raw_reply)
        return TicketResponse(**parsed_json)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Model output was not valid JSON.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))