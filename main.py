import requests
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

ROBOFLOW_URL = "https://serverless.roboflow.com/tapp-workspace/workflows/small-object-detection-sahi"

def build_payload(image: str):
    return {
        "api_key": os.environ["ROBOFLOW_API_KEY"],
        "inputs": {
            "image": {
                "type": "base64",
                "value": image
            },
            "video": {
                "video_identifier": "dummy-video-1",
                "frame_number": 0,
                "frame_timestamp": "2026-06-17T00:00:00Z",
                "fps": 30,
                "measured_fps": 30,
                "comes_from_video_file": False
            }
        }
    }

class InferenceRequest(BaseModel):
    image: str
    use_cache: bool = False

@app.post("/infer")
async def run_inference(req: InferenceRequest):
    try:
        response = requests.post(ROBOFLOW_URL, json=build_payload(req.image), timeout=30)
        response.raise_for_status()
        return {"success": True, "result": response.json()}
    except requests.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"{e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer-debug")
async def run_inference_debug(req: InferenceRequest):
    try:
        response = requests.post(ROBOFLOW_URL, json=build_payload(req.image), timeout=30)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}