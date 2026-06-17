import os
import base64
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference_sdk import InferenceHTTPClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY environment variable not set")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


class InferenceRequest(BaseModel):
    image: str
    use_cache: bool = False


def clean_base64_image(image: str) -> str:
    if image.startswith("data:image"):
        image = image.split(",", 1)[1]
    return image.strip()


@app.post("/infer")
async def run_inference(req: InferenceRequest):
    temp_path = None

    try:
        image_data = clean_base64_image(req.image)

        image_bytes = base64.b64decode(image_data)

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = client.run_workflow(
            workspace_name="tapp-workspace",
            workflow_id="small-object-detection-sahi",
            images={
                "image": temp_path
            },
            use_cache=req.use_cache
        )

        print("=== ROBOFLOW SDK RESPONSE ===")
        print(result)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/infer-debug")
async def run_inference_debug(req: InferenceRequest):
    temp_path = None

    try:
        image_data = clean_base64_image(req.image)

        image_bytes = base64.b64decode(image_data)

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = client.run_workflow(
            workspace_name="tapp-workspace",
            workflow_id="small-object-detection-sahi",
            images={
                "image": temp_path
            },
            use_cache=False
        )

        return {
            "base64_length": len(image_data),
            "decoded_length": len(image_bytes),
            "temp_file_exists": os.path.exists(temp_path),
            "temp_file_size": os.path.getsize(temp_path),
            "response": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_present": bool(API_KEY)
    }