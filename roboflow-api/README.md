# Roboflow Inference API

A small FastAPI backend that proxies requests to Roboflow's serverless inference,
keeping your API key safe on the server side.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API will be at http://localhost:8000

## Deploy to Render (free tier)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo
4. Set the environment variable `ROBOFLOW_API_KEY` in Render's dashboard
5. Deploy — Render auto-detects `render.yaml`

## Calling the API from your GitHub Pages frontend

```javascript
const toBase64 = file => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result.split(",")[1]);
  reader.onerror = reject;
  reader.readAsDataURL(file);
});

async function runInference(imageFile) {
  const base64 = await toBase64(imageFile);
  const response = await fetch("https://your-app.onrender.com/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: base64 })
  });
  return response.json();
}
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/infer` | Run workflow on a base64 image |
| GET | `/health` | Health check |
