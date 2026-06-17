import os
import json
import requests

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

ROBOFLOW_URL = (
    "https://serverless.roboflow.com/"
    "tapp-workspace/workflows/small-object-detection-sahi"
)

session = requests.Session()


class InferenceRequest(BaseModel):
    image: str
    use_cache: bool = False


def clean_base64_image(image: str) -> str:
    """
    Removes browser-style data URI prefixes if present.

    Example:
    data:image/jpeg;base64,/9j/4AAQ...
    ->
    /9j/4AAQ...
    """
    if image.startswith("data:image"):
        image = image.split(",", 1)[1]

    return image.strip()


def build_payload(image: str):
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable not set")

    image = clean_base64_image(image)

    return {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image
            }
        }
    }


@app.post("/infer")
async def run_inference(req: InferenceRequest):
    try:
        payload = build_payload(req.image)

        response = session.post(
            ROBOFLOW_URL,
            json=payload,
            timeout=30,
        )

        try:
            result = response.json()
        except Exception:
            result = {
                "raw_response": response.text
            }

        print("\n=== ROBOFLOW REQUEST ===")
        print(f"Status: {response.status_code}")

        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)

        if not response.ok:
            raise HTTPException(
                status_code=response.status_code,
                detail=result
            )

        return {
            "success": True,
            "status_code": response.status_code,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/infer-debug")
async def run_inference_debug(req: InferenceRequest):
    try:
        payload = build_payload(req.image)

        response = session.post(
            ROBOFLOW_URL,
            json=payload,
            timeout=30,
        )

        try:
            result = response.json()
        except Exception:
            result = response.text

        return {
            "request_url": ROBOFLOW_URL,
            "request_payload": payload,
            "status_code": response.status_code,
            "response": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "roboflow_url": ROBOFLOW_URL,
        "api_key_present": bool(os.getenv("ROBOFLOW_API_KEY"))
    }