from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference_sdk import InferenceHTTPClient
import os

app = FastAPI()

# Allow requests from your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your GitHub Pages URL in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"]  # Set this in your hosting env vars
)

class InferenceRequest(BaseModel):
    image: str  # base64-encoded image string
    use_cache: bool = True

@app.post("/infer")
async def run_inference(req: InferenceRequest):
    try:
        result = client.run_workflow(
            workspace_name="tapp-workspace",
            workflow_id="small-object-detection-sahi",
            images={"image": req.image},
            use_cache=req.use_cache
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
