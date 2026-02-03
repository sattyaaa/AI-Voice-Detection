import base64
import os
import contextlib
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [AudioShield] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our detection engine
from detect import detector

# load_dotenv() - REMOVED per user request for Render deployment

# LIFESPAN MANAGER (Resolves Cold Start)
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Logic
    logger.info("--- Warming up the AI engine... ---")
    try:
        # Trigger model loading in a threadpool so it doesn't block startup completely if async
        # But here we want to block until ready.
        # Run a dummy inference to ensure weights are on device
        dummy_audio = b'\x00' * 16000 # 1 sec silent
        await run_in_threadpool(detector.analyze_audio, dummy_audio, "English")
        logger.info("--- AI Engine Ready & Warmed Up! ---")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
    
    yield
    
    # Shutdown Logic
    logger.info("--- Shutting down AudioShield ---")

app = FastAPI(
    title="AudioShield AI: Voice Fraud Detector",
    version="2.0",
    docs_url="/docs",
    lifespan=lifespan
)

# CONFIGURATION
# Default key from problem statement example: sk_test_123456789
VALID_API_KEY = os.getenv("API_KEY", "sk_test_123456789")

# MODELS (Strict Adherence to Spec)
class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Language: Tamil, English, Hindi, Malayalam, Telugu")
    audioFormat: str = Field(..., pattern="^(?i)mp3$", description="Must be 'mp3'")
    audioBase64: str = Field(..., description="Base64 encoded MP3 audio")

class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str # AI_GENERATED or HUMAN
    confidenceScore: float
    explanation: str

# ROUTES
@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
async def detect_voice(
    request: VoiceDetectionRequest
):
    # 1. Security Check - REMOVED for Public Access per user request
    # logger.info(f"Public Access: Processing request for {request.language}")

    try:
        # 2. Basic Validation (Logic)
        if request.audioFormat.lower() != "mp3":
            # Just to be perfectly safe, though Pydantic regex handles it
            raise ValueError("Only MP3 format is supported.")

        # 3. Decode Base64
        try:
            audio_data = base64.b64decode(request.audioBase64)
        except Exception:
            raise ValueError("Invalid Base64 encoding.")
            
        if not audio_data:
             raise ValueError("Empty audio data.")

        # 4. Perform Detection (Non-Blocking)
        # We run the synchronous detector.analyze_audio in a threadpool
        # so the API remains responsive to other requests.
        logger.info(f"Processing request for language: {request.language}")
        
        result = await run_in_threadpool(detector.analyze_audio, audio_data, request.language)

        if "error" in result:
             # If internal analysis failed, we still want to return a strict error format if possible,
             # or map it to the error response.
             raise ValueError(result["error"])

        # 5. Return formatted response (Strict JSON)
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=result["classification"],
            confidenceScore=result["confidenceScore"],
            explanation=result["explanation"]
        )

    except ValueError as ve:
        logger.error(f"Validation Error: {ve}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(ve)}
        )
    except Exception as e:
        logger.error(f"Internal Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error processing audio."}
        )

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "AudioShield AI (Hackathon Edition)",
        "models_loaded": len(detector.pipelines) if hasattr(detector, 'pipelines') else 0
    }

# Standard execution for HF Spaces (uvicorn launched via Docker CMD)
