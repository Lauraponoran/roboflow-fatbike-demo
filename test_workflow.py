import base64
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="S0cz7iDI1fFXIDuHYKWv"  # use a freshly regenerated key
)

# Encode the local image as base64, the same way your FastAPI backend
# expects to receive it from the frontend.
image_path = "fatbike.jpg"  # e.g. "./test.jpg"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")

dummy_video_metadata = {
    "video_identifier": "dummy-video-1",
    "frame_number": 0,
    "frame_timestamp": "2026-06-17T00:00:00Z",
    "fps": 30,
    "measured_fps": 30,
    "comes_from_video_file": False
}

result = client.run_workflow(
    workspace_name="tapp-workspace",
    workflow_id="small-object-detection-sahi",
    images={"image": encoded_image},
    parameters={"video": dummy_video_metadata},
    use_cache=False
)

print(result)
