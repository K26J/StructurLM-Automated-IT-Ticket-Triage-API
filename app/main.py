import os
import json
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel

# 1. Define what the API expects to receive from the customer
class TicketRequest(BaseModel):
    subject: str
    message: str

# 2. Setup a storage dictionary so the heavy AI model stays in memory
ml_models = {}

# 3. Safety switch for local testing (so we don't crash your PC)
IS_LOCAL_TEST = os.getenv("TEST_MODE", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """This function runs once when the server starts up."""
    
    # If we are just testing the API locally, skip loading the heavy AI model
    if IS_LOCAL_TEST:
        print("LOCAL TEST MODE: Skipping heavy model load to save your RAM.")
        yield
        return

    # 4. Define where our model files are located
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_folder = "./model/qwen2.5_ticket_lora_upgraded"
    
    # Check if we have a GPU (CUDA) or just a standard CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print("GPU found! Loading the 4-bit compressed model...")
        # Compress the model so it fits easily in the GPU memory
        compress_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=compress_config,
            device_map="auto"
        )
    else:
        # BULLETPROOF SAFETY FOR YOUR 8GB PC
        raise RuntimeError(
            "\n--- HARDWARE SAFETY TRIGGERED ---\n"
            "No GPU detected and TEST_MODE is off.\n"
            "Loading this model on your CPU will crash your 8GB PC.\n"
            "Please run this command instead: TEST_MODE=true uvicorn app.main:app\n"
            "---------------------------------"
        )
    
    print("Attaching the fine-tuned LoRA brain to the base model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_folder)
    model = PeftModel.from_pretrained(base_model, adapter_folder)
    
    # Speed up the model settings for generating text
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    
    # Create a simple pipeline to handle the heavy lifting
    ml_models["pipeline"] = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    print("Model is fully loaded and ready to read tickets!")
    
    # Keep the server running
    yield 
    
    # Clean up memory when the server is shut down
    ml_models.clear() 

# 5. Start the actual FastAPI web server
app = FastAPI(title="Ticket Sorting AI API", lifespan=lifespan)

@app.post("/predict")
def predict_ticket(request: TicketRequest):
    """This function runs every time a customer submits a ticket."""
    
    # If testing locally, just return fake data instantly
    if IS_LOCAL_TEST:
        return {
            "department_queue": "IT Support",
            "urgency": "High",
            "ticket_type": "Incident",
            "primary_tag": "App Crash",
            "_note": "This is a fake test response!"
        }

    # 6. Give the AI its strict instructions
    instructions = (
        "You are an AI assistant that classifies support tickets. "
        "Extract the department_queue, urgency, ticket_type, and primary_tag. "
        "Output strictly valid JSON with no markdown formatting."
    )
    
    # Combine the subject and message into one clean string
    customer_ticket = f"Subject: {request.subject}\nMessage: {request.message}"
    
    # Format the conversation for the AI
    chat_history = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": customer_ticket}
    ]
    
    try:
        # 7. Ask the AI to generate the answer
        output = ml_models["pipeline"](
            chat_history,
            max_new_tokens=50,
            do_sample=False, # Tell the AI to be exact, not creative
            return_full_text=False,
            pad_token_id=ml_models["pipeline"].tokenizer.eos_token_id
        )
        
        # 8. Clean up the AI's raw text and convert it into a real JSON dictionary
        raw_ai_text = output[0]["generated_text"].strip()
        return json.loads(raw_ai_text)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="The AI messed up the JSON formatting.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))